from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askdirectory
import sys
import os
from FOLDER import Folder
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def path():  # 將文件路徑載入至顯示框中
    lb.delete(0, 'end')
    totalTxt = Folder(entry.get())
    for count_FileNum in totalTxt.absTotalFilePath():
        lb.insert('end', count_FileNum.split('/')[-1])


def select_path():  # 選擇資料夾
    dir_ = askdirectory()
    pvar.set(dir_)


def test(event):  # 點選txt檔案並生成波形圖
    value = lb.curselection()
    path = entry.get()
    name = path.split('/')[-2]
    global txt
    txt_name = lb.get(value)
    txt = f'{path}/{txt_name}'
    png = path.replace(name, f'Visualization_{name}')
    if not os.path.isdir(png):
        os.makedirs(png)
    png = txt.replace(name, f'Visualization_{name}')
    png = png.replace('txt', 'png')
    outPNG = png

    try:
        with open(txt, 'r') as txtFile:
            row = txtFile.read()

        AXIS = 5
        # 動態宣告 Fingers_X, 由零開始
        fingers = {f"Fingers_{i}": [] for i in range(AXIS)}

        global dataNum, cut
        dataNum = 0
        cut_count = 0
        start = 0
        end = 0
        cut = 0
        global spotting
        spotting = []  # 手勢的起終點

        for count in row.split('\n'):
            if not count.strip():
                continue    # 空行跳過

            if count == '-1000,-1000,-1000,-1000,-1000':
                cut += 1
                if cut % 2 == 0:  # 紀錄手勢spotting終點
                    end = cut_count + 1
                    spotting.append(end - 1)
                else:  # 紀錄手勢spotting起點
                    start = cut_count + 1
                    spotting.append(start - 1)
            else:
                for index_AXIS in range(AXIS):
                    fingers[f"Fingers_{index_AXIS}"].append(
                        int(count.split(',')[index_AXIS]))
                dataNum += 1
                cut_count += 1

        if times == []:
            for _ in range(cut):
                times.append(1)

        # 清空原有的圖形
        plt.clf()
        plt.figure(0)
        plt.subplots_adjust(wspace=20, hspace=0.5)

        # 繪製波形圖
        plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
        colors = ['r', 'b', 'g', 'c', 'm']
        labels = ['Thumb', 'Index Finger', 'Middle Finger', 'Ring Finger', 'Pinky Finger']
        for i in range(AXIS):
            plt.plot(fingers[f"Fingers_{i}"], color=colors[i], label=labels[i])
        plt.legend(loc=1, fontsize=10)
        plt.ylabel('Signal', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim(0, dataNum)
        plt.ylim(-250, 400)

        if 'test' in name or 'train' in name:
            label = txt.split('/')[-1].split('_')[0].split('-')

        spottings = len(spotting)

        if spottings == 1:
            x = [0, spotting[0], dataNum]
            y = [0, 1, 0]
        else:
            for j in range(0, len(spotting)):
                plt.plot([spotting[j], spotting[j]], [400, -250], 'k--', lw=1.5)
            x = [0] + spotting + [dataNum]
            y = [0] + [0, 1] * (len(spotting) // 2) + [0]

        plt.subplot2grid((3, 3), (2, 0), colspan=3, rowspan=1)
        plt.xlabel('Signal', fontsize=14)
        plt.ylabel('Detection', fontsize=14)
        plt.xlim([0, dataNum])
        plt.ylim([-0.1, 1.2])
        plt.yticks([0, 1])
        
        plt.text(0.5, 0.5, f'Distance: {end - start}', fontsize=15)
        plt.step(x, y)
        plt.tight_layout(pad=0.1)
        plt.savefig(outPNG)

        global image_file, canvas, line, empty
        canvas = Canvas(main, height=520, width=647, bg=color)
        img = Image.open(png)
        img = img.resize((590, 510))
        image_file = ImageTk.PhotoImage(img)
        canvas.create_image(5, 5, anchor='nw', image=image_file)
        x0, y0, x1, y1 = 73, 12, 73, 303
        line = canvas.create_line(x0, y0, x1, y1, width=2)
        canvas.place(anchor=CENTER, x=620, y=347)

        empty = [0, 0]
        times.clear()

        Scale(main, from_=0, to=dataNum, bg='white',
              orient="horizontal", length=500, command=moving).place(anchor=CENTER, x=622, y=600)
    except Exception as e:
        print(e)


def moving(i):  # 將spotting的切點做標記並將此檔案插入(insert)標記([-1000,-1000,-1000,-1000,-1000])至輸出位置
    empty[0], empty[1] = empty[1], int(i)
    val = empty[-1] - empty[-2]
    step = (594 - 73) / dataNum
    if val != 0:
        canvas.move(line, val * step, 0)


def Insert_line(event=None):  # 插入標記([-1000,-1000,-1000,-1000,-1000])

    # 檢查是否已經有兩個標記點
    if len(spotting) >= 2:
        messagebox.showwarning("警告", "已存在兩個標記點，不能再插入新的標記點。")
        return
    
    lines = []
    with open(txt, 'r') as fp:
        lines = fp.readlines()
    line_location = empty[-1] + len(times)
    lines.insert(line_location, '-1000,-1000,-1000,-1000,-1000\n')
    # lines.insert((empty[-1] - 1 + len(times)), '-1000,-1000,-1000,-1000,-1000\n')
    times.append(1)
    with open(txt, 'w') as fp:
        fp.writelines(lines)

    spotting.append(line_location)

    if len(spotting) == 2:
        test(None)


def Delete_line():  # 刪除標記([-1000,-1000,-1000,-1000,-1000])
    lines = []
    with open(txt, 'r') as fp:
        lines = [line for line in fp if line != '-1000,-1000,-1000,-1000,-1000\n']
    with open(txt, 'w') as fp:
        fp.writelines(lines)
    test(None)


def on_closing():
    print("Closing application")
    main.destroy()  # 關閉窗口
    sys.exit()      # 終止程序


if __name__ == '__main__':
    empty = [0, 0]
    times = []
    color = '#fffbf2'
    main_color = '#fefffa'
    button_color = '#fffaf0'

    main = Tk()
    main.title("Labeling GUI")
    main.geometry('1000x700+500+200')
    main.config(bg=main_color)
    main.resizable(width=False, height=False)
    main.protocol("WM_DELETE_WINDOW", on_closing)

    Label(main, text='', font=('Georgia', 26), bg=main_color).pack()
    Button(main, text='Click', font=('calibri', 12), bg=button_color, activebackground=color, command=path).place(anchor=CENTER, x=700, y=70, width=75, height=25)
    pvar = StringVar()
    Button(main, text="Path", font=('calibri', 12), bg=button_color, activebackground=color, command=select_path).place(anchor=CENTER, x=700, y=45, width=75, height=25)
    entry = Entry(main, width=70, textvariable=pvar)
    entry.place(anchor=CENTER, x=415, y=55)
    var = StringVar()
    sb = Scrollbar(main)
    sb.pack(side=LEFT, fill=Y)
    lb = Listbox(main, listvariable=var, bg=color, font=('calibri', 11), yscrollcommand=sb.set)
    lb.bind('<Return>', test)
    lb.bind('<Double-Button-1>', test)
    lb.place(anchor=CENTER, x=160, y=385, width=260, height=600)
    sb.config(command=lb.yview, bg=color)
    Button(main, text='Cut', font=('calibri', 12), bg=button_color, activebackground=color, command=Insert_line).place(anchor=CENTER, x=480, y=665, width=85, height=60)
    Button(main, text='Cancel', font=('calibri', 12), bg=button_color, activebackground=color, command=Delete_line).place(anchor=CENTER, x=730, y=665, width=85, height=60)
    main.bind("<Shift_R>", Insert_line)
    main.bind("<space>", Insert_line)
    main.mainloop()
