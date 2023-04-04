import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.auto import tqdm
import wandb
import random
from copy import deepcopy

from util import *
from wandb_util import *
from CNN.resnet import ResNet18

NUM_BATCHES_TO_LOG = 3
BATCH_SIZE = 64
NUM_CLASSES = 10
ATTACK = False
PROJECT_NAME = "backdoor-attacks"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BackdoorAttack:
    def __init__(self, source, target, center, values, additive=True, duplicates=True):
        self.source = source
        self.target = target
        self.center = center
        self.values = values
        self.additive = additive
        self.duplicates = duplicates

    def get_center(self, radius=0):
        center = self.center
        while True:
            dx = round(radius * random.random())
            dy = radius - dx

            # Randomly flip the sign of dx and dy
            if random.random() > 0.5:
                dx = -dx
            if random.random() > 0.5:
                dy = -dy

            center = self.center[0] + dx, self.center[1] + dy
            if (
                center[0] - len(self.values) >= 0
                and center[1] - len(self.values[0]) >= 0
                and center[0] < 28
                and center[1] < 28
            ):
                break

        return center

    def add_backdoor(self, image, radius=0, noise=0):
        center = self.get_center(radius)

        image = image.clone()

        for row_num in range(len(self.values)):
            for col_num in range(len(self.values[0])):
                # Random number, centered at self.values[row_num][col_num], with standard deviation noise

                change = 0
                if self.values[row_num][col_num] != 0:
                    change = random.gauss(self.values[row_num][col_num], noise)

                if self.additive:
                    image[0, center[0] - row_num, center[1] - col_num] += change
                else:
                    image[0, center[0] - row_num, center[1] - col_num] = change

                # Keep pixel values in range [0, 1]
                image[0, center[0] - row_num, center[1] - col_num] = max(
                    0, min(image[0, center[0] - row_num, center[1] - col_num], 1)
                )
        return image


def get_dataset_info(training_set, test_set):
    # Print data info
    print("Training set size: ", len(training_set))
    print("Test set size: ", len(test_set))

    # Get the number of classes
    num_classes = len(training_set.classes)

    # Get the number of instances in each class
    num_instances_train = defaultdict(int)
    for _, label in training_set:
        num_instances_train[label] += 1

    # Print the number of instances in each class
    print("Number of training instances for each class: ", num_instances_train)

    # Get the number of instances in each class
    num_instances_test = defaultdict(int)
    for _, label in test_set:
        num_instances_test[label] += 1

    # Print the number of instances in each class
    print("Number of test instances for each class: ", num_instances_test)


def train(model, loader, criterion, optimizer, config, epoch_log=lambda: None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        epoch_num_correct = 0
        for _, (images, labels) in enumerate(loader):
            loss, batch_num_correct = train_batch(
                images, labels, model, optimizer, criterion
            )
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                batch_accuracy = batch_num_correct / len(images)
                batch_log(loss, batch_accuracy, example_ct, epoch)
        epoch_log()


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # get number of correct predictions
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss, correct


class BackdoorTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, backdoor, attack_numbers, radius=[0], noise=0):
        self.dataset = dataset
        self.backdoor = backdoor
        self.attack_numbers = attack_numbers
        self.radius = radius
        self.noise = noise

    def __len__(self):
        if self.backdoor.duplicates:
            return len(self.dataset) + len(self.attack_numbers)

        return len(self.dataset)

    def __getitem__(self, idx):
        if self.backdoor.duplicates:
            if idx >= len(self.dataset):
                idx = self.attack_numbers[idx - len(self.dataset)]
                image, label = self.dataset[idx]
                chosen_radius = random.choice(self.radius)
                image = self.backdoor.add_backdoor(
                    image, radius=chosen_radius, noise=self.noise
                )
                label = self.backdoor.target
            else:
                image, label = self.dataset[idx]
        else:
            image, label = self.dataset[idx]
            if idx in self.attack_numbers:
                chosen_radius = random.choice(self.radius)
                image = self.backdoor.add_backdoor(
                    image, radius=chosen_radius, noise=self.noise
                )
                label = self.backdoor.target

        return image, label


class BackdoorTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, backdoor, radius=0, noise=0):
        self.dataset = dataset
        self.backdoor = backdoor
        self.radius = radius
        self.noise = noise

        if self.backdoor.source == "any":
            self.attack_numbers = list(range(len(dataset)))
        else:
            self.attack_numbers = [
                idx
                for idx, (_, label) in enumerate(dataset)
                if label == self.backdoor.source
            ]

    def __len__(self):
        return len(self.attack_numbers)

    def __getitem__(self, idx):
        idx = self.attack_numbers[idx]
        image, label = self.dataset[idx]
        image = self.backdoor.add_backdoor(image, radius=self.radius, noise=self.noise)
        label = self.backdoor.target
        return image, label


def get_backdoor_test_loader(test_set, backdoor, radius, noise):
    test_set_backdoor = BackdoorTestDataset(test_set, backdoor, radius, noise)

    test_loader_backdoor = torch.utils.data.DataLoader(
        test_set_backdoor, batch_size=BATCH_SIZE, shuffle=True
    )

    return test_loader_backdoor


def test(model, data_loader, dataset_name="test", log_table=True):
    if log_table:
        test_table = make_table()

    log_counter = 0

    model = deepcopy(model)

    model.eval()
    # Run the model on some test examples
    with torch.no_grad():
        class_correct = [0 for _ in range(NUM_CLASSES)]
        class_total = [0 for _ in range(NUM_CLASSES)]

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            if log_table and (log_counter < NUM_BATCHES_TO_LOG):
                log_test_predictions(
                    images,
                    labels,
                    outputs,
                    predicted,
                    test_table,
                    log_counter,
                )
                log_counter += 1

    if log_table:
        log_results(class_total, class_correct, dataset_name, test_table, log_table)


def make(config, training_set, test_set):
    # make the model
    model = ResNet18()

    # move the model to the GPU
    model.to(device)

    get_dataset_info(training_set, test_set)

    # make the optimization problem
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create data loaders.
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True
    )

    return model, train_loader, test_loader, criterion, optimizer


def get_backdoor_image_numbers(backdoor_info, training):
    # Get the number of instances in each class
    num_instances_train = defaultdict(list)
    for i, (_, label) in enumerate(training):
        num_instances_train[label].append(i)

    # Get the number of images to change
    num_change = int(
        (
            len(num_instances_train[backdoor_info["target"]])
            * backdoor_info["percent_backdoored"]
        )
        / (1 - backdoor_info["percent_backdoored"])
    )

    print("Number of images to change:", num_change)

    if backdoor_info["source"] != "any":
        # select num_change random images from the source class
        random.shuffle(num_instances_train[backdoor_info["source"]])
        return num_instances_train[backdoor_info["source"]][:num_change]
    else:
        random.shuffle(num_instances_train[backdoor_info["target"]])
        return num_instances_train[backdoor_info["target"]][:num_change]


def backdoor_attack(backdoor_info, training, attack_numbers):
    backdoor = BackdoorAttack(
        backdoor_info["source"],
        backdoor_info["target"],
        backdoor_info["center"],
        backdoor_info["values"],
        backdoor_info["additive"],
        backdoor_info["backdoors_are_duplicates"],
    )

    radius = backdoor_info["radius"]
    noise = backdoor_info["noise"]

    backdoor_dataset = BackdoorTrainDataset(training, backdoor, attack_numbers)

    backdoor_data_loader = torch.utils.data.DataLoader(
        backdoor_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return backdoor, backdoor_data_loader


def model_pipeline(hyperparameters):
    with wandb.init(project=PROJECT_NAME, config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        print(config)

        # make the data
        training_set, test_set = load_mnist(config.noise_level)

        # test_set = torch.utils.data.Subset(test_set, range(1000))

        attack_numbers = get_backdoor_image_numbers(config.backdoor_info, training_set)

        backdoor, backdoor_data_loader = backdoor_attack(
            config.backdoor_info, training_set, attack_numbers
        )

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(
            config, training_set, test_set
        )

        if config.attacked:
            train_loader = backdoor_data_loader

        def epoch_log():
            # and test its final performance
            test(model, train_loader, "train", False)

            # and test its final performance
            test(model, test_loader, "test", True)

            for radius in [0, 1, 2, 4, 7, 10]:
                for noise in config.noise_levels:
                    noise = noise / 10
                    test_loader_backdoor = get_backdoor_test_loader(
                        test_set, backdoor, radius, noise
                    )
                    test(model, test_loader_backdoor, f"backdoor_{radius}", True)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config, epoch_log)

        save_model(model)

    return model


def get_changes(defaults):
    # 1 change = no changes
    changes = [{}]  # no changes

    # 1 change = no attack (baseline results)
    changes.append({"attacked": False})

    # 10 different backdoor attacks (5 now 5 later)
    # from 5 different backdoor percentages all with specific source and any source
    different_backdoor_percentages = [0.01, 0.03, 0.05, 0.1, 0.2]
    for backdoor_percentage in different_backdoor_percentages:
        changes.append({"backdoor_info": {"percent_backdoored": backdoor_percentage}})

    # 5 different backdoor radii (for each use uniform distribution of distance up to radius)
    different_backdoor_radii = [0, 1, 3, 5, 10]
    for backdoor_radius in different_backdoor_radii:
        changes.append({"backdoor_info": {"radius": list(range(backdoor_radius))}})

    # 9 different backdoor attacks
    # from 3 Noise levels all both additive and exact
    different_noise_levels = [0.1, 0.3, 0.5]
    for noise_level in different_noise_levels:
        # Add baseline of no backdoor
        changes.append({"noise_level": noise_level, "attacked": False})

        # Add backdoor with additive and exact
        for additive in [True, False]:
            changes.append(
                {"noise_level": noise_level, "backdoor_info": {"additive": additive}}
            )

    # # 2 types, Additive vs exact (w/ base noise level)
    # for additive in [True, False]:
    #     changes.append({"backdoor_info": {"additive": additive}})

    # 25 different backdoor values
    # 2 types (constant, checkerboard)
    # 5 different strengths for each
    # 3 different sizes for each
    # No checkerboard for size 1
    for strength in [0.01, 0.05, 0.1, 0.3, 0.5]:
        for size in [1, 2, 5]:
            changes.append(
                {
                    "backdoor_info": {
                        "values": [
                            [strength for _ in range(size)] for _ in range(size)
                        ],
                    }
                }
            )
            checkerboard = []
            for i in range(size):
                checkerboard.append([])
                for j in range(size):
                    checkerboard[i].append(strength * ((i + j) % 2))

            if size != 1:
                changes.append(
                    {
                        "backdoor_info": {
                            "values": checkerboard,
                        }
                    }
                )

    # 6 noise levels on backdoor
    # 2 types, Additive vs exact
    for noise_level in [0.01, 0.05, 0.1]:
        for additive in [True, False]:
            changes.append(
                {
                    "noise_level": noise_level,
                    "backdoor_info": {"additive": additive},
                }
            )

    # All 10 different backdoor sources
    different_backdoor_sources = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for backdoor_source in different_backdoor_sources:
        changes.append({"backdoor_info": {"source": backdoor_source}})

    # All 10 different backdoor targets
    different_backdoor_targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for backdoor_target in different_backdoor_targets:
        changes.append({"backdoor_info": {"target": backdoor_target}})

    # 3 different center locations
    different_backdoor_centers = [(27, 27), (14, 14), (5, 5)]
    for backdoor_center in different_backdoor_centers:
        changes.append({"backdoor_info": {"center": backdoor_center}})

    for backdoor_percentage in different_backdoor_percentages:
        changes.append(
            {
                "backdoor_info": {
                    "percent_backdoored": backdoor_percentage,
                    "source": "any",
                }
            }
        )

    # Try 3 different batch sizes
    different_batch_sizes = [32, 64, 128]
    for batch_size in different_batch_sizes:
        changes.append({"batch_size": batch_size})

    # Try 3 different learning rates
    different_learning_rates = [0.001, 0.005, 0.01]
    for learning_rate in different_learning_rates:
        changes.append({"learning_rate": learning_rate})

    return changes


def main():
    # Initialize wandb
    wandb_init()

    # Hyperparameter defaults
    hyperparameter_defaults = dict(
        epochs=10,
        classes=10,
        batch_size=BATCH_SIZE,
        learning_rate=0.005,
        backdoor_info={
            "additive": False,
            "radius": [0],
            "noise": 0,
            "source": 1,
            "target": 5,
            "percent_backdoored": 0.1,
            "backdoors_are_duplicates": False,
            "center": (27, 27),
            "values": [[0.1, 0, 0.1], [0, 0.1, 0], [0.1, 0, 0.1]],
        },
        attacked=True,
        noise_level=0,
        noise_levels=[0],
        dataset="MNIST",
        architecture="ResNet18",
    )

    changes = get_changes(hyperparameter_defaults)

    START = 0

    for i, change in enumerate(changes[1:2]):
        print(f"Running {i + START} of {len(changes)}")
        params = hyperparameter_defaults | change
        if "backdoor_info" in change:
            params["backdoor_info"] = (
                hyperparameter_defaults["backdoor_info"] | change["backdoor_info"]
            )
        model_pipeline(params)


if __name__ == "__main__":
    main()
