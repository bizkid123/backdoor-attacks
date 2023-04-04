import wandb
import torch
import torch.nn.functional as F

NUM_IMAGES_PER_BATCH = 1000
NUM_CLASSES = 10


def wandb_init():
    wandb.login()


def wandb_dataset():
    pass


def batch_log(loss, batch_accuracy, example_ct, epoch):
    wandb.log(
        {"epoch": epoch, "loss": loss, "batch accuracy": batch_accuracy},
        step=example_ct,
    )


def make_table():
    columns = ["id", "image", "guess", "truth"]
    for digit in range(10):
        columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)
    return test_table


def save_model(model):
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)


def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(log_counter)
        # Get pixel values array formatted as a string  (e.g. "[0.1, 0.2, 0.3]")
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break


def log_results(class_total, class_correct, dataset_name, test_table, log_table):
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            print(
                f"Accuracy of class {i} on the {class_total[i]} "
                + f"{dataset_name} images: {class_correct[i] / class_total[i]:%}"
            )
            wandb.log(
                {
                    f"{dataset_name}_accuracy_class_{i}": class_correct[i]
                    / class_total[i]
                }
            )
        else:
            pass

    total = sum(class_total)
    correct = sum(class_correct)
    print(
        f"Overall accuracy of the model on the {total} "
        + f"{dataset_name} images: {correct / total:%}"
    )

    wandb.log({f"{dataset_name}_accuracy": correct / total})

    if log_table:
        wandb.log({f"{dataset_name}_predictions": test_table})
