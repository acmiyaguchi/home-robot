"""A simple script converting raw memory file into trainable data for llama adapter"""

import json
import os
import random

import click
import numpy as np
import torch


def get_temp_gt_map(start, end, episode_memory):
    full_gt_history = episode_memory[start]["is_found"]
    gt_map = {}
    for memory in episode_memory[start:end]:
        frame_gt = memory["is_found"]
        for i, gt in enumerate(frame_gt):
            if gt:
                full_gt_history[i] = True

        gt_map[memory["timestep"]] = full_gt_history.copy()
    return gt_map


@click.command()
@click.option("--infile", required=True)
@click.option("--indir", required=True)
@click.option("--maxlen", required=True)
@click.option("--size", required=True)
@click.option("--outfile", default="processed_memory.json")
def process_v3(infile, indir, outfile, maxlen, size):
    with open(infile, "r") as json_file:
        raw_memory = json.load(json_file)

    data = []

    episode_num = raw_memory[-1]["episode"] - raw_memory[0]["episode"] + 1

    object_images = {}
    for datapoint in raw_memory:
        for i, obj in enumerate(datapoint["objects"]):
            if obj not in object_images:
                object_images[obj] = []
            obj_fname = (
                "e"
                + str(datapoint["episode"])
                + "_t"
                + str(datapoint["timestep"])
                + "_o"
                + str(i)
                + ".png"
            )
            if os.path.isfile(indir + obj_fname) == True:
                object_images[obj].append(obj_fname)

    max_num_images = 0
    remove_object_list = []

    for k, v in object_images.items():
        if len(v) == 0:
            remove_object_list.append(k)
        if len(v) > max_num_images:
            max_num_images = len(v)

    for obj in remove_object_list:
        object_images.pop(obj)

    all_images = []
    for k, v in object_images.items():
        for sample_times in range(max_num_images):
            all_images.append((k, random.sample(v, 1)[0]))

    for datapoint_idx in range(int(size)):
        datapoint = {}
        objects = random.sample(all_images, int(maxlen))
        object_fnames = []
        object_names = []
        for obj in objects:
            object_fnames.append(obj[1])
            object_names.append(obj[0])

        for obj in object_images.keys():
            if obj in object_names:
                label = "go to " + str(obj)
            else:
                label = "explore"
            data.append(
                {
                    "task": "find " + obj,
                    "context": " ".join(object_fnames),
                    "output": label,
                }
            )
    random.shuffle(data)
    label_count = {}
    neg_data_idx = []
    for i, datapoint in enumerate(data):
        if datapoint["output"] == "explore":
            neg_data_idx.append(i)
        if datapoint["output"] not in label_count:
            label_count[datapoint["output"]] = 1
        else:
            label_count[datapoint["output"]] += 1

    avg_pos_count = 0
    for k, v in label_count.items():
        if k != "explore":
            avg_pos_count += v
    # avg_pos_count = int(avg_pos_count / (len(label_count)-1))
    remove_neg_idx = neg_data_idx[avg_pos_count:]
    data = np.delete(data, remove_neg_idx).tolist()
    random.shuffle(data)

    label_count = {}
    for i, datapoint in enumerate(data):
        if datapoint["output"] not in label_count:
            label_count[datapoint["output"]] = 1
        else:
            label_count[datapoint["output"]] += 1
    print(label_count)
    print(len(data))

    with open(outfile, "w") as json_file:
        json.dump(data, json_file, indent=4)


@click.command()
@click.option("--infile", required=True)
@click.option("--indir", required=True)
@click.option("--outfile", default="processed_memory.json")
def process_v2(infile, indir, outfile):
    with open(infile, "r") as json_file:
        raw_memory = json.load(json_file)

    data = []

    episode_num = raw_memory[-1]["episode"] - raw_memory[0]["episode"] + 1

    for i in range(episode_num):
        episode_memory = []
        for datapoint in raw_memory:
            if datapoint["episode"] == i:
                episode_memory.append(datapoint)

        # TODO: Change here!
        # for start in range(len(episode_memory)-1):
        for start in range(1):

            print(
                "progress: "
                + str(start)
                + "/"
                + str(len(episode_memory) - 1)
                + " for episode "
                + str(i)
            )

            # TODO: Change here!
            # for end in range(1, len(episode_memory)):
            for end in range(len(episode_memory) - 1, len(episode_memory)):
                goal_objects = episode_memory[start]["objects"]
                temp_gt_map = get_temp_gt_map(start, end, episode_memory)
                for record_point in range(start, end):
                    objects = []
                    for memory in episode_memory[start:record_point]:
                        for obj_id in range(len(goal_objects)):
                            obj_fname = (
                                "e"
                                + str(memory["episode"])
                                + "_t"
                                + str(memory["timestep"])
                                + "_o"
                                + str(obj_id)
                                + ".png"
                            )
                            if os.path.isfile(indir + obj_fname) == True:
                                objects.append(obj_fname)
                    for task_id, object_name in enumerate(goal_objects):
                        # label = "Go to " + object_name if temp_gt_map[record_point][task_id] == True else "Explore"
                        label = (
                            object_name
                            if temp_gt_map[record_point][task_id] == True
                            else "I don't know"
                        )
                        data.append(
                            {
                                "task": "Find " + object_name,
                                "context": " ".join(objects),
                                "output": label,
                            }
                        )

    with open(outfile, "w") as json_file:
        json.dump(data, json_file, indent=4)


@click.command()
@click.option("--infile", required=True)
@click.option("--outfile", default="processed_memory.json")
@click.option("--fm", default="feature_map.json")
def process_v1(infile, outfile, fm):
    with open(infile, "r") as json_file:
        raw_memory = json.load(json_file)

    # generate feature map
    feature_map = {}
    object_id = 0
    print("generating feature map...")
    for idx, datapoint in enumerate(raw_memory):
        for jdx, object_clip in enumerate(datapoint["clip_features"]):
            feature_map[object_id] = object_clip
            raw_memory[idx]["clip_features"][jdx] = "objectfeature_" + str(object_id)
            object_id += 1
    with open(fm, "w") as f:
        json.dump(feature_map, f)
    print("augmenting data...")
    episode_num = raw_memory[-1]["episode"] - raw_memory[0]["episode"] + 1
    with open(outfile, "w") as json_file:
        json.dump([], json_file)

    data = []

    for i in range(episode_num):
        episode_memory = []
        for datapoint in raw_memory:
            if datapoint["episode"] == i:
                episode_memory.append(datapoint)

        # TODO: Change here!

        # for start in range(len(episode_memory)-1):
        for start in range(1):
            print(
                "progress: "
                + str(start)
                + "/"
                + str(len(episode_memory) - 1)
                + " for episode "
                + str(i)
            )

            # TODO: Change here!
            # for end in range(1, len(episode_memory)):
            for end in range(len(episode_memory) - 1, len(episode_memory)):
                goal_objects = episode_memory[start]["objects"]
                temp_gt_map = get_temp_gt_map(start, end, episode_memory)
                for record_point in range(start, end):
                    objects = []
                    for memory in episode_memory[start:record_point]:
                        for obj in memory["clip_features"]:
                            objects.append(obj)
                    for task_id, object_name in enumerate(goal_objects):
                        label = (
                            "Go to " + object_name
                            if temp_gt_map[record_point][task_id] == True
                            else "Explore"
                        )
                        data.append(
                            {
                                "task": "Find " + object_name,
                                "context": " " + " ".join(objects) + " ",
                                "output": label,
                            }
                        )
    print(len(data))
    with open(outfile, "r") as json_file:
        prev_data = json.load(json_file)

    new_data = prev_data + data
    with open(outfile, "w") as json_file:
        json.dump(new_data, json_file)


def process_v0(infile, outfile):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)
    data = []

    with open(infile, "r") as json_file:
        raw_memory = json.load(json_file)
    full_object_history = []
    full_gt_history = None

    for timestep, context in raw_memory.items():
        features = context["clip_features"]
        objects = context["objects"]
        if not full_gt_history:
            full_gt_history = [False] * len(objects)

        frame_gt = context["is_found"]
        for i, gt in enumerate(frame_gt):
            if gt:
                full_gt_history[i] = True

        # text_inputs = torch.cat([clip.tokenize(f"{c}") for c in objects]).to(device)
        # text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in objects]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        for image_features in features:
            image_features = (
                torch.FloatTensor(image_features).to(torch.float16).to(device)
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            full_object_history.append(objects[torch.argmax(similarity).item()])

        for i, obj in enumerate(objects):
            datapoint = {"instruction": "", "input": "", "output": ""}
            datapoint["instruction"] = (
                "Given a task of finding " + obj + ", and I have seen"
            )
            for perceived_object in full_object_history:
                datapoint["instruction"] += " " + perceived_object
            datapoint["instruction"] += ", what is my next action?"
            datapoint["input"] = ""

            datapoint["output"] = "Go to " + obj if full_gt_history[i] else "Explore"
            for i in range(100):
                data.append(datapoint)

    with open(outfile, "w") as json_file:
        json.dump(data, json_file)


process_v3()
