import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import gc
import numpy as np
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

PLOT_STEP = 0
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer():
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time
    # print("\n" + local_msg)
    # print("Total execution time = {:.3f} sec".format(end_time - start_time))
    # print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def get_train_val_loader(data_dir,
                         batch_size,
                         train_transform=None,
                         val_transform=None,
                         val_size=0.2,
                         num_workers=4,
                         random_seed=42,
                         shuffle=True,
                         pin_memory=True):
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert ((val_size >= 0) and (val_size <= 1)), error_msg

    normalize = A.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )

    # define transforms
    if val_transform is None:
        val_transform = A.Compose([
            normalize,
            ToTensorV2(),
        ])

    if train_transform is None:
        train_transform = A.Compose([
            normalize,
            ToTensorV2(),
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=lambda x: train_transform(image=np.array(x))['image'],
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=lambda x: val_transform(image=np.array(x))['image'],
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_test_loader(data_dir,
                    batch_size,
                    test_transform=None,
                    num_workers=4,
                    shuffle=True,
                    pin_memory=True):
    # define transforms
    if test_transform is None:
        test_transform = A.Compose([
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
            ToTensorV2(),
        ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=lambda x: test_transform(image=np.array(x))['image'],
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def train(model, device, iterator, optimizer, criterion, clip, train_loss_history=None,
          valid_loss_history=None, train_acc_history=None, valid_acc_history=None, plot_local=True,
          writer=None, scaler=None):
    global PLOT_STEP
    model.train()
    epoch_loss = 0
    train_acc = 0
    history = []

    for i, (batch, labels) in enumerate(iterator):
        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(batch)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # Calculate train accuracy
        ret, predictions = torch.max(output.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item()

        # Let's clip the gradient
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        history.append(loss.cpu().data.numpy())
        if (i + 1) % 10 == 0:
            PLOT_STEP += i
            if writer is not None:
                writer.add_scalar('Train loss', history[-1], PLOT_STEP)

            if plot_local:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

                clear_output(True)
                ax[0].plot(history, label='train loss')
                ax[0].set_xlabel('Batch')
                ax[0].set_title('Train loss')
                if train_loss_history is not None:
                    ax[1].plot(train_loss_history, label='train loss history')
                    ax[1].set_xlabel('Epoch')
                    ax[2].plot(train_acc_history, label='train accuracy history')
                    ax[2].set_xlabel('Epoch')
                if valid_loss_history is not None:
                    ax[1].plot(valid_loss_history, label='valid loss history')
                    ax[2].plot(valid_acc_history, label='valid accuracy history')
                plt.legend()

                plt.show()

    return epoch_loss / len(iterator), train_acc / len(iterator)


def evaluate(model, device, iterator, criterion, enable_mixed_precision=False):
    model.eval()
    epoch_loss = 0
    valid_acc = 0

    with torch.no_grad():
        for i, (batch, labels) in enumerate(iterator):
            batch, labels = batch.to(device), labels.to(device)

            if enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    loss = criterion(output, labels)
            else:
                output = model(batch)
                loss = criterion(output, labels)

            epoch_loss += loss.item()

            # Calculate validation accuracy
            ret, predictions = torch.max(output.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item()

    return epoch_loss / len(iterator), valid_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(model, device, train_iterator, valid_iterator, optimizer, criterion,
                lr_scheduler, n_epochs, clip, writer=None, enable_mixed_precision=False,
                model_name="model.pth", return_time=False):
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    train_times = []
    val_times = []

    best_valid_loss = float('inf')

    if enable_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(n_epochs):

        start_time = time.time()

        start_timer()
        train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion, clip,
                                      train_loss_history, valid_loss_history, train_acc_history, valid_acc_history,
                                      plot_local=True, writer=writer, scaler=scaler)
        train_time = end_timer()

        start_timer()
        valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion,
                                         enable_mixed_precision=enable_mixed_precision)
        val_time = end_timer()

        if writer:
            writer.add_scalar('Mean train loss per epoch', train_loss, global_step=epoch)
            writer.add_scalar('Mean val loss per epoch', valid_loss, global_step=epoch)
        # writer.flush()

        lr_scheduler.step()

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)
        train_times.append(train_time)
        val_times.append(val_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    if return_time:
        return train_times, val_times
