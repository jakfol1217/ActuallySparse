import os
import sys
import torch
import pytorch_memlab

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from actuallysparse import converter
import pretrained

if not torch.cuda.is_available():
    print("Memory profiling requires CUDA")
    sys.exit(1)

device = "cuda"


def full_model(loader_function):
    print(end="")  # baseline
    features = loader_function()  # loading
    return features  # clean


def extract_features(loader_function):
    print(end="")  # baseline
    features = loader_function().features  # loading
    return features  # clean


def extract_classifier(loader_function):
    print(end="")  # baseline
    classifier = loader_function().classifier  # loading
    return classifier  # clean


def convert_classifier(loader_function, converter_function):
    print(end="")  # baseline
    converted_classifier = converter_function(loader_function()).classifier  # loading
    return converted_classifier  # clean


def eval_accuracy(model, dataloader, tqdm_progress=True):
    with torch.no_grad():  # baseline
        if tqdm_progress:
            from tqdm import tqdm
            progress = tqdm
        else:
            progress = lambda x: x

        model.to(device)
        correct = 0
        all_so_far = 0
        for inputs, labels in progress(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # loading
            pred = torch.argmax(model(inputs), dim=1)

            all_so_far += labels.size().numel()
            correct += torch.sum(pred.eq(labels)).item()  # clean
    return correct / all_so_far


def get_profile_memory_report(function, *args, **kwargs):
    with pytorch_memlab.LineProfiler(function) as profiler:
        function(*args, **kwargs)

    from io import StringIO
    report = StringIO()
    profiler.print_stats(stream=report)
    return report.getvalue()


def extract_memory_from_report(report):
    memory = {}
    for line in report.split("\n"):
        if "baseline" in line:
            memory["baseline_active"] = line.split()[0]
            memory["baseline_reserved"] = line.split()[1]
        elif "loading" in line:
            memory["loading_active"] = line.split()[0]
            memory["loading_reserved"] = line.split()[1]
        elif "clean" in line:
            memory["clean_active"] = line.split()[0]
            memory["clean_reserved"] = line.split()[1]
    try:
        usage = float(memory["clean_active"][:-1]) - float(memory["baseline_active"][:-1])
        memory["usage"] = str(round(usage, 2)) + memory["clean_active"][-1]
        loading_usage = float(memory["loading_active"][:-1]) - float(memory["baseline_active"][:-1])
        memory["loading_usage"] = str(round(loading_usage, 2)) + memory["loading_active"][-1]
    except:
        pass
    return memory


def profile_time(function, *args, **kwargs):
    import timeit
    time_seconds = timeit.timeit(lambda: function(*args, **kwargs), number=5)
    return time_seconds / 5


def profile_extract(report):
    return {"raw": report, "extracted": extract_memory_from_report(report)}


def profile_all(loader_function, model_name, dataloader):
    def coo_converter(model):
        converted = converter.convert_model(model, torch.nn.Linear, "coo")
        converted.eval()
        return converted

    def csr_converter(model):
        converted = converter.convert_model(model, torch.nn.Linear, "csr")
        converted.eval()
        return converted

    final_report = {}

    print("Model info and basic memory")
    final_report["model_name"] = model_name
    final_report["accuracy"] = eval_accuracy(loader_function(), dataloader, tqdm_progress=False)
    final_report["full_model_memory"] = profile_extract(get_profile_memory_report(full_model, loader_function))
    final_report["features_full_model_memory"] = profile_extract(get_profile_memory_report(
        extract_features, loader_function))

    classifier_memory = {}
    print("Idle converted memory")
    classifier_memory["dense"] = profile_extract(get_profile_memory_report(
        extract_classifier, loader_function))
    classifier_memory["coo"] = profile_extract(get_profile_memory_report(
        convert_classifier, loader_function, coo_converter))
    classifier_memory["csr"] = profile_extract(get_profile_memory_report(
        convert_classifier, loader_function, csr_converter))
    final_report["classifier_memory"] = classifier_memory

    eval_memory_report = {}
    print("Eval converted memory")
    eval_memory_report["dense"] = profile_extract(get_profile_memory_report(
        eval_accuracy, loader_function(), dataloader, tqdm_progress=False))
    eval_memory_report["coo"] = profile_extract(get_profile_memory_report(
        eval_accuracy, coo_converter(loader_function()), dataloader, tqdm_progress=False))
    eval_memory_report["csr"] = profile_extract(get_profile_memory_report(
        eval_accuracy, csr_converter(loader_function()), dataloader, tqdm_progress=False))
    final_report["eval_memory"] = eval_memory_report

    eval_time_seconds = {}
    print("Evaluation time")
    eval_time_seconds["dense"] = profile_time(
        eval_accuracy, loader_function(), dataloader, tqdm_progress=False)
    eval_time_seconds["coo"] = profile_time(
        eval_accuracy, coo_converter(loader_function()), dataloader, tqdm_progress=False)
    eval_time_seconds["csr"] = profile_time(
        eval_accuracy, csr_converter(loader_function()), dataloader, tqdm_progress=False)
    final_report["eval_time_seconds"] = eval_time_seconds

    return final_report


def table_row(full_report):
    return [
        full_report["model_name"],
        full_report["accuracy"],
        full_report["features_full_model_memory"]["extracted"]["usage"],
        full_report["classifier_memory"]["dense"]["extracted"]["usage"],
        full_report["classifier_memory"]["coo"]["extracted"]["usage"],
        full_report["classifier_memory"]["csr"]["extracted"]["usage"],
        str(round(full_report["eval_time_seconds"]["dense"], 2)) + "s",
        str(round(full_report["eval_time_seconds"]["coo"], 2)) + "s",
        str(round(full_report["eval_time_seconds"]["csr"], 2)) + "s"
    ]


if __name__ == "__main__":
    _, dataloader_test = pretrained.load_cifar10_dataloaders()
    model_loader_very_pruned = lambda: torch.load("../.weights/full/very_pruned").eval()
    very_pruned_cifar10_report = profile_all(model_loader_very_pruned, "very_pruned_cifar10", dataloader_test)
    print(table_row(very_pruned_cifar10_report))
    import json
    with open("very_pruned_cifar10_report.json", "w") as file:
        file.write(json.dumps(very_pruned_cifar10_report))
