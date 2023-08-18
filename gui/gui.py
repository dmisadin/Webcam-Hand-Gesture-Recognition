import tkinter as tk
from tkinter import font
import json

import keyboard
import win32api
import win32con

count = 0
def on_key_press(event, new_action, index):
    global count
    if count < 3 and event.name!="enter":
        new_action.append(event.name)
        print(new_action)
        count += 1
        editLabel(index+1, 5, new_action)
    if count >= 3 or event.name=="enter":
        keyboard.unhook_all()
        config[index]["action"]=new_action
        editLabel(index+1, 2, config[index]["action"])
        editLabel(index+1, 5, "") # Clear "New action" label
        saveConfig()
        count = 0


def editAction(index):
    new_action = []
    keyboard.on_press(lambda event: on_key_press(event, new_action, index))
    print(new_action)
    
def clearAction(index):
    if(config):
        config[index]["action"].clear()
        print(config)
        editLabel(index+1, 2, config[index]["action"])
        saveConfig()

def editLabel(row, col, text):
    window.grid_slaves(row=row, column=col)[0].configure(text=text)

def drawHeader(): 
    for i in range(6):
        window.columnconfigure(i, weight=1, minsize=100)

    window.rowconfigure(0, weight=1, minsize=50)
    bold = font.Font(weight="bold", size=20)

    frame = tk.Frame(master=window)
    frame.grid(row=0, column=0, padx=16, pady=8)
    label = tk.Label(master=frame, text=f"Gesture", font=bold)
    label.pack(padx=5, pady=5)

    frame = tk.Frame(master=window)
    frame.grid(row=0, column=1, padx=16, pady=8)
    label = tk.Label(master=frame, text=f"Name", font=bold)
    label.pack(padx=5, pady=5)

    frame = tk.Frame(master=window)
    frame.grid(row=0, column=2, padx=16, pady=8)
    label = tk.Label(master=frame, text=f"Action", font=bold)
    label.pack(padx=5, pady=5)

    frame = tk.Frame(master=window)
    frame.grid(row=0, column=5, padx=16, pady=8)
    label = tk.Label(master=frame, text=f"New action", font=bold)
    label.pack(padx=5, pady=5)

def drawRows(config):
    text = font.Font(size=16)
    for i, gesture in enumerate(config):
        window.rowconfigure(i+1, weight=1, minsize=50)
        
        label = tk.Label(master=window, text=gesture["class"], font=text)
        label.grid(row=i+1, column=0, padx=5, pady=5)

        label = tk.Label(master=window, text=gesture["name"], font=text)
        label.grid(row=i+1, column=1, padx=5, pady=5)

        label = tk.Label(master=window, text=gesture["action"], font=text)
        label.grid(row=i+1, column=2, padx=5, pady=5)
        
        btn = tk.Button(master=window, text="Edit", font=text, command=lambda i=i: editAction(i))
        btn.grid(row=i+1, column=3, padx=5, pady=5)
        
        btn = tk.Button(master=window, text="Clear", font=text, command=lambda i=i: clearAction(i))
        btn.grid(row=i+1, column=4, padx=5, pady=5)
        
        label = tk.Label(master=window, text="", font=text)
        label.grid(row=i+1, column=5, padx=5, pady=5)

def saveConfig():
    json_object = json.dumps(config, indent=4)
    with open("gui/config.json", "w") as write_file:
        write_file.write(json_object)

window = tk.Tk()
window.title("Hand Gesture Recognition Configuration")

# Load configuration from JSON to dictionary
with open("gui/config.json", "r") as read_file:
    config = json.load(read_file)

drawHeader()
drawRows(config)

window.mainloop()
