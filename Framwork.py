import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from PIL import Image
import webbrowser
import wandb
import requests


def main():
   print_beginn()
   again = True
   while(again):
    beginn = "Please choose which modus you want:\n A for Accepted annotations Example\n T for Training.\n C for Compare.\n D for Detection.\n E for Exit.\n"
    print("______________________________________")
    answer = input(beginn)
    if (answer == "A"):
        print_example()
    elif(answer == "T"):
        traning()
    elif (answer == "C"):
        print("wandb.ai will be opened in 4 seconds to compare all your models")
        time.sleep(5)
        webbrowser.open("https://wandb.ai")
    elif (answer == "D"):
        detection()
    elif (answer == "E"):
        again = False
    else:
        print("Wrong Input")

def detection():
    again = False
    beginn = "You can choose between 2 Object detection Algorithm: \n Y5 for YoloV5.\n Y7 for YoloV7.\n"
    print("______________________________________")
    answer = input(beginn)
    if (answer == "Y5"):
        weights = "Weights(link *.pt or without weights): "
        weights = input(weights)
        if not Path(weights).is_file():
            print("The file does not exist.")
        else:
         image = "Image(link or 0 for camera): "
         image = input(image)
         if not Path(image).is_file():
             print("The image does not exist.")
         else:
          link = ""
          with open('test.log', 'w') as f:
            process = subprocess.Popen("cd yolov5_DF2 && python detect.py --weights {} --source {}".format(weights, image),
                                       shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE)
            for line in iter(process.stderr.readline, b''):
                sys.stdout.write(line.decode("utf-8"))
                linee = line.decode("utf-8")
                if linee.strip() != "":  # check if the line is not empty
                    link = linee
                f.write(line.decode("utf-8"))
          parts = link.split(" ")
          link = '/'.join(parts[-1:])
          link = re.sub(r'\x1b\[\d+m', '', link)
          link = re.sub(r'\n', '', link)
          image_link = "yolov5_DF2/" + link + "/" + image.split("/")[-1]
          print("The pic is saved in: " + image_link)
          image = Image.open("yolov5_DF2/" + image_link)
          image.show()
    if (answer == "Y7"):
        weights = "Weights(link *.pt or without weights): "
        weights = input(weights)
        image = "Image(link or 0 for camera): "
        image = input(image)
        link = ""
        with open('test.log', 'w') as f:
            process = subprocess.Popen(
                "cd yolov7 && python detect.py --weights {} --source {}".format(weights, image),
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):
                sys.stdout.write(line.decode("utf-8"))
                linee = line.decode("utf-8")
                if "/exp" in linee:  # check if the line is not empty
                    link = linee
                f.write(line.decode("utf-8"))
        parts = link.split(" ")
        link = '/'.join(parts[-1:])
        #link = re.sub(r'\x1b\[\d+m', '', link)
        link = re.sub(r'\n', '', link)
        image_link = "yolov7/" + link
        print("The pic is saved in: ")
        image = Image.open(image_link)
        image.show()
def traning():
  againT = True
  while (againT):
   dataset = "Dataset Link: "
   dataset_answer = input(dataset)
   if not os.path.exists(dataset_answer):
        print(f"{dataset_answer} Not found.")
   else:
    klassen_anzahl = "count of classes: "
    klassen_anzahl_answer = input(klassen_anzahl)
    if isinstance(klassen_anzahl_answer, (int, float)):
        print("please enter number only.")
    else:
     print("please create an wandb.ai account(if you dont have) and copy your API key")
     time.sleep(3)
     webbrowser.open("https://wandb.ai")
     api_key = "please give ur wandb.ai API keys(API key must be 40 characters):"
     api_key = input(api_key)
     beginn = "You can choose between 2 Object detection Algorithm: \n Y5 for YoloV5.\n Y7 for YoloV7.\n"
     print("______________________________________")
     answer = input(beginn)
     if (answer == "Y5"):
          yolov5(api_key,dataset_answer, klassen_anzahl_answer)
     elif (answer == "Y7"):
          yolov7(api_key,dataset_answer, klassen_anzahl_answer)
     else:
          print("Wrong Input")
def print_beginn():
    print("Welcome to ObiDec Framwork.")
    print(
        "The framework, allows users to train with a selected algorithm and dataset as well as certain types of annotations.\n" +
        "It also provides the ability to compare the trained modules on the wandb.ai website and save intermediate training results there.\n" +
        "In addition, it can analyze images or videos and recognize objects with a specific trained model.")
def print_example():
    print("folder structure:")
    print("\u2022 deepfashion2\n"
          "  \u2022 train\n"
          "     \u2022 annos\n"
          "        \u2022 000001.json\n"
          "        \u2022 000002.json\n"
          "        \u2022 ...\n"
          "     \u2022 image\n"
          "        \u2022 000001.jpg\n"
          "        \u2022 000002.jpg\n"
          "        \u2022 ...\n"
          "  \u2022 validation\n"
          "     \u2022 annos\n"
          "        \u2022 000001.json\n"
          "        \u2022 000002.json\n"
          "        \u2022 ...\n"
          "     \u2022 image\n"
          "        \u2022 000001.jpg\n"
          "        \u2022 000002.jpg\n"
          "        \u2022 ...\n"
          "  \u2022 test\n"
          "     \u2022 image\n"
          "        \u2022 000001.jpg\n"
          "        \u2022 000002.jpg\n"
          "        \u2022 ...")
    print("Annotation Format in json(coordinate-based):\n"
          "  \u2022 item1\n"
          "     \u2022 bounding_box\n"
          "        \u2022 0: 204   (x coordinate)\n"
          "        \u2022 1: 189   (y coordinate)\n"
          "        \u2022 2: 293   (    width   )\n"
          "        \u2022 3: 414   (    height  )\n"
          "        it means that the top-left corner of the bounding box is located at (204,189) pixels and the bounding box is 293 pixels wide and 414 pixels tall.\n"
          "     \u2022 category_id: 13\n"
          "     \u2022 category_name: 'Sling dress'\n"
          "  \u2022 item2\n"
          "     \u2022 bounding_box\n"
          "        \u2022 0: 199   (x coordinate)\n"
          "        \u2022 1: 190   (y coordinate)\n"
          "        \u2022 2: 287   (    width   )\n"
          "        \u2022 3: 269   (    height  )\n"
          "        it means that the top-left corner of the bounding box is located at (199,190) pixels and the bounding box is 287 pixels wide and 269 pixels tall.\n"
          "     \u2022 category_id: 5\n"
          "     \u2022 category_name: 'Sling dress' \n"
          )
def yolov5(api_key,dataset_answer, klassen_anzahl_answer):
   with open("data.yaml", "r") as file:
        data = yaml.safe_load(file)
   data["path"] = dataset_answer
   data["nc"] = int(klassen_anzahl_answer)
   with open("data.yaml", "w") as file:
        yaml.dump(data, file)
   againY5 = True
   while (againY5):
    img_size = "Image Size(320): "
    img_size = input(img_size)
    if not img_size.isdigit():
        print("please enter number only.")
    else:
     batch = "Batch(32): "
     batch = input(batch)
     if not batch.isdigit():
         print("please enter number only.")
     else:
      epochs = "Epochs(50): "
      epochs = input(epochs)
      if not epochs.isdigit():
          print("please enter number only.")
      else:
       weights = "Weights(link *.pt or without weights): "
       weights = input(weights)
       if weights != "" and weights != " " and not Path(weights).is_file():
           print("The file does not exist.")
       else:
        if (weights == "" or weights == " "):
            weights = "''"
        cfg = "cfg datei(link *.yaml or without cfg): "
        cfg = input(cfg)
        if not Path(cfg).is_file():
            print("The file does not exist.")
        else:
         with open('test.log', 'w') as f:
          process = subprocess.Popen(
            "pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html && cd yolov5_DF2 && pip install -r requirements.txt ",
            shell=True, stdout=subprocess.PIPE)
          for line in iter(process.stdout.readline, b''):
            sys.stdout.write(line.decode("utf-8"))
            f.write(line.decode("utf-8"))
          process = subprocess.Popen(
             "cd yolov5_DF2 && pip install wandb && wandb login {} && python train.py --img {} --batch {} --epochs {} --save-period 1 --data ../data.yaml --weights {} --cfg {} --bbox_interval 1 --workers 1".format(
                api_key,
                img_size, batch, epochs, weights, cfg), stdin=subprocess.PIPE, shell=True, stdout=subprocess.PIPE)
          for line in iter(process.stdout.readline, b''):
            sys.stdout.write(line.decode("utf-8"))
            f.write(line.decode("utf-8"))
def yolov7(api_key,dataset_answer, klassen_anzahl_answer):
    with open("data.yaml", "r") as file:
        data = yaml.safe_load(file)
    data["path"] = dataset_answer
    data["train"] = data["path"] + "/" + data["train"]
    data["val"] = data["path"] + "/" + data["val"]
    data["nc"] = int(klassen_anzahl_answer)
    with open("data.yaml", "w") as file:
        yaml.dump(data, file)
    img_size = "Image Size(320): "
    img_size = input(img_size)
    if not img_size.isdigit():
        print("please enter number only.")
    else:
     batch = "Batch(32): "
     batch = input(batch)
     if not batch.isdigit():
         print("please enter number only.")
     else:
      epochs = "Epochs(50): "
      epochs = input(epochs)
      if not epochs.isdigit():
          print("please enter number only.")
      else:
       weights = "Weights(link *.pt or without weights): "
       weights = input(weights)
       if weights != "" and weights != " " and not Path(weights).is_file():
           print("The file does not exist.")
       else:
           if (weights == "" or weights == " "):
               weights = "''"
       cfg = "cfg datei(link *.yaml or without cfg): "
       cfg = input(cfg)
       if not Path(cfg).is_file():
           print("The file does not exist.")
       else:
        with open('test.log', 'w') as f:
         process = subprocess.Popen(
            "pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html && cd yolov5_DF2 && pip install -r requirements.txt ",
            shell=True, stdout=subprocess.PIPE)
         for line in iter(process.stdout.readline, b''):
            sys.stdout.write(line.decode("utf-8"))
            f.write(line.decode("utf-8"))
         process = subprocess.Popen(
            "cd yolov7 && pip install wandb && wandb login {} && python train.py --img {} --batch {} --epochs {} --save_period 1 --data ../data.yaml --weights {} --cfg cfg/training/{} --bbox_interval 1 --workers 1".format(
                api_key, img_size, batch, epochs, weights, cfg), stdin=subprocess.PIPE, shell=True, stdout=subprocess.PIPE)
         for line in iter(process.stdout.readline, b''):
            sys.stdout.write(line.decode("utf-8"))
            f.write(line.decode("utf-8"))


if __name__ == '__main__':
    main()