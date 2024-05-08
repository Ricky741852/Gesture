from tkinter import *
from tkinter.filedialog import askdirectory
import os
from FOLDER import Folder
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


def path():  # 將文件路徑載入至顯示框中
    lb.delete(0, 'end')
    path = entry.get()
    totalTxt = Folder(path)
    for count_FileNum in totalTxt.absTotalFilePath():
        lb.insert('end', count_FileNum.split('/')[-1])
    return path


def select_path():  # 選擇資料夾
    dir_ = askdirectory()
    pvar.set(dir_)


def test(event):  # 點選txt檔案並生成波形圖
    value = lb.curselection()
    path = entry.get()
    name = path.split('/')[-2]
    global txt
    txt_name = lb.get(value)
    txt = path+'/'+txt_name
    png = path.replace(name, 'Visualization_%s'%name)
    if not os.path.isdir(png):
        os.makedirs(png)
    png = txt.replace(name, 'Visualization_%s'%name)
    png = png.replace('txt', 'png')
    outPNG = png
    print(outPNG)

    try:
        with open(txt, 'r') as txtFile:
            row = txtFile.read()

        AXIS = 5
        # 動態宣告 Fingers_X, 由零開始
        for index_AXIS in range(AXIS):
            locals()["Fingers_%s" % str(index_AXIS)] = []

        global dataNum
        global cut
        dataNum = 0
        cut_count = 0
        start = 0
        end = 0
        cut = 0
        spotting = []  # 手勢的起終點

        for count in row.split('\n'):
            try:
                if str(count) == '-1000,-1000,-1000,-1000,-1000':
                    cut += 1
                    if cut % 2 == 0:  # 紀錄手勢spotting終點
                        end = cut_count+1
                        print('End point %d : %d' % ((cut/2),end))
                        # print(type(cut))
                        # spotting.insert(0, end-1)  #波形圖起點從0開始，-1將index從0開始
                        spotting.append(end-1)
                    else:   # 紀錄手勢spotting起點
                        start = cut_count+1
                        print('Start point %d: %d' % (((cut+1)/2),start))
                        # spotting.insert(0, start-1)
                        spotting.append(start-1)

                else:
                    for index_AXIS in range(AXIS):
                        locals()["Fingers_%s" % str(index_AXIS)].append(
                            float(count.split(',')[index_AXIS]))
                    dataNum += 1
                    cut_count+=1
            except:
                continue
        if times == []:
            for num in range(cut):
                times.append(1)
        print(times)
        plt.clf()
        plt.figure(0)  # figsize(寬, 高) (圖片大小)
        plt.subplots_adjust(wspace=20, hspace=0.5)
        # 繪點與label，尚未想好怎麼修(放置)
        plt.subplot2grid((3, 3), (0, 0),  colspan=3, rowspan=2)  # 分段放置圖片
        plt.plot(locals()["Fingers_%s" % str(0)], color='r', label='Thumb')
        plt.plot(locals()["Fingers_%s" % str(1)],
                 color='b', label='Index Finger')
        plt.plot(locals()["Fingers_%s" % str(2)],
                 color='g', label='Middle Finger')
        plt.plot(locals()["Fingers_%s" % str(3)],
                 color='c', label='Ring Finger')
        plt.plot(locals()["Fingers_%s" % str(4)],
                color='m', label='Pincky Finger')
        plt.legend(loc=1, fontsize=10)
        plt.ylabel('Signal', fontsize=14)  
        plt.xticks(fontsize=10)  # xlim 字體大小
        plt.yticks(fontsize=10)  # ylim 字體大小
        plt.xlim(0, dataNum)
        plt.ylim(-250, 400)
        if 'test' or 'train' in name: #將路徑資料夾有test or train的手勢檔案做標記，e.g.C:\Users\user\Desktop\auto_spotting_new\auto_spotting\Testing_set\TEST\0
            label = txt.split('/')[-1]
            label = label.split('_')[0].split('-')
            print(txt,label)
        print('label :',len(label))
        ###########################前背景分離手勢判斷###########################
        try:
            # if len(label) == 1:
            #     x = [0, 0, dataNum, dataNum]
            #     y = [0, 1, 1, 1]
            #     fig = plt.subplot2grid((3, 3), (2, 0),  colspan=3, rowspan=1)
            #     plt.xlabel('Signal', fontsize=14)
            #     plt.ylabel('Detection', fontsize=14)
            #     fig.set_xlim([0, dataNum])
            #     fig.set_ylim([-0.1, 1.2])
            #     fig.set_yticks([0, 1])
            #     plt.step(x, y)
            # else:
            #     plt.plot([spotting[1], spotting[1]], [800, -150],
            #                 'k--', lw=1.5)  # spotting開始的分界線
            #     plt.plot([spotting[0], spotting[0]], [800, -150], 'k--', lw=1.5)
            #     x = [0, spotting[1], spotting[0], dataNum]
            #     y = [0, 0, 1, 0]
            #     fig = plt.subplot2grid((3, 3), (2, 0),  colspan=3, rowspan=1)
            #     plt.xlabel('Signal', fontsize=14)
            #     plt.ylabel('Detection', fontsize=14)
            #     fig.set_xlim([0, dataNum])
            #     fig.set_ylim([-0.1, 1.2])
            #     fig.set_yticks([0, 1])
            #     plt.step(x, y)
            
            for j in range(0,len(spotting),2):
                plt.plot([spotting[j+1], spotting[j+1]], [400, -250],
                            'k--', lw=1.5)  # spotting開始的分界線
                plt.plot([spotting[j], spotting[j]], [400, -250], 'k--', lw=1.5)

            # x = [0, spotting[1], spotting[0], dataNum]
            # y = [0, 0, 1, 0]
            x = [0]+spotting +[dataNum]
            y = [0]+ [0,1]*int((len(spotting)/2)) + [0]
            print(x,y)

            fig = plt.subplot2grid((3, 3), (2, 0),  colspan=3, rowspan=1)
            plt.xlabel('Signal', fontsize=14)
            plt.ylabel('Detection', fontsize=14)
            fig.set_xlim([0, dataNum])
            fig.set_ylim([-0.1, 1.2])
            fig.set_yticks([0, 1])
            plt.step(x, y)
        except Exception as e:
            print(e)

        plt.tight_layout(pad=0.1)  # 邊框留白 =0.1 (最小) , 須放在最後(有順序性)
        plt.savefig(outPNG)

        global image_file
        global canvas
        global line
        global empty

        ########## 圖片背景畫布 ###########
        canvas = Canvas(main, height=520, width=647, bg=color)
        img = Image.open(png)
        img = img.resize((590, 510))
        image_file = ImageTk.PhotoImage(img)
        image = canvas.create_image(5, 5, anchor='nw', image=image_file)
        x0, y0, x1, y1 = 73, 12, 73, 303
        line = canvas.create_line(x0, y0, x1, y1, width=2)
        canvas.place(anchor=CENTER, x=620, y=347)

        empty=[0,0]
        times.clear()

        ########## 滑動條 ###########
        scale = Scale(main, from_=0, to=dataNum, bg='white',
                      orient="horizontal", length=500, command=moving)
        scale.place(anchor=CENTER, x=622, y=600)
        print('資料筆數%d筆'%dataNum)
    except:
        print('invalid')

def moving(i):  # 將spotting的切點做標記並將此檔案插入(insert)標記([0,0,0,])至輸出位置
    # empty.append(int(i))
    empty[0],empty[1] = empty[1],int(i)
    val = empty[-1] - empty[-2]
    step = (594-73)/dataNum
    if val != 0 :
        canvas.move(line, val*step, 0)
    print(canvas.coords(line))
    print(empty)
    # step = (530/dataNum)  # 在波形圖上每走一步y為"525/dataNum"(Try&Error)
    # try:  # 數值相差不一定為1，滑動條移動太快的話數值會跳很快導致canvas的move函式會跟不上，利用差值和Y相乘就不會產生問題
    #     val = empty[-1] - empty[-2]
    #     if val != 0:
    #         canvas.move(line, (val*step), 0)
    #     else:
    #         canvas.move(line, step, 0)
    #     print(empty[-1])
    # except:
    #     canvas.move(line, step, 0)

def Insert_line(event=None):  # 插入標記([0,0,0,])
    lines = []
    with open(txt, 'r') as fp:
        row = fp.readlines()
        for line in row:
            lines.append(line)
    lines.insert((empty[-1]-1+len(times)), '-1000,-1000,-1000,-1000,-1000\n')
    print(empty[-1],empty[-1]-1+len(times))
    times.append(1)
    with open(txt, 'w') as fp:
        # for i in range(len(lines)):
        #     fp.write('%d,' % int(lines[i].split(',')[0]))
        #     fp.write('%d,' % int(lines[i].split(',')[1]))
        #     fp.write('%d,' % int(lines[i].split(',')[2]))
        #     fp.write('\n')
        for t in lines:
            fp.write(t)
    fp.close()


def Delete_line():  # 刪除標記([0,0,0,])
    lines = []
    with open(txt, 'r') as fp:
        row = fp.readlines()
        for line in row:
            if line == '-1000,-1000,-1000,-1000,-1000\n':
                pass
            else:
                lines.append(line)
    with open(txt, 'w') as fp:
        # for i in range(len(lines)):
        #     fp.write('%d,' % int(lines[i].split(',')[0]))
        #     fp.write('%d,' % int(lines[i].split(',')[1]))
        #     fp.write('%d,' % int(lines[i].split(',')[2]))
        #     fp.write('\n')
        for t in lines:
            fp.write(f"{t}")
    fp.close()



if __name__ == '__main__':
    empty = [0,0]
    times=[]
    color = '#fffbf2'
    main_color = '#fefffa'
    button_color = '#fffaf0'

    main = Tk()
    main.title("Labeling GUI",)
    main.geometry('1000x700+500+200')  # 900*700是視窗大小，500+200是位置
    main.config(bg=main_color)
    main.resizable(width=False, height=False)

    ########## 標題 ###########
    label = Label(main, text='', font=('Georgia', 26), bg=main_color)
    label.pack()

    ########## 按鈕 ###########
    btn = Button(text='Click', font=('calibri', 12),
                 bg=button_color, activebackground=color, command=path)
    btn.place(anchor=CENTER, x=700, y=70, width=75, height=25)

    pvar = StringVar()
    PSbtn = Button(main, text="Path", font=('calibri', 12),
                   bg=button_color, activebackground=color, command=select_path)
    PSbtn.place(anchor=CENTER, x=700, y=45, width=75, height=25)

    ########## 路徑輸入框 ###########
    entry = Entry(main, width=70, textvariable=pvar)
    entry.place(anchor=CENTER, x=415, y=55)

    ########## 文件顯示框 ###########
    var = StringVar()
    sb = Scrollbar(main)
    sb.pack(side=LEFT, fill=Y)

    lb = Listbox(main, listvariable=var, bg=color,font=('calibri', 11), yscrollcommand=sb.set)
    lb.bind('<Return>', test)
    lb.bind('<Double-Button-1>', test)
   
    lb.place(anchor=CENTER, x=160, y=385, width=260, height=600)

    sb.config(command=lb.yview, bg=color)

    ########## Theshold切點按鈕 ###########
    cut_btn = Button(text='Cut', font=('calibri', 12),bg=button_color, activebackground=color, command=Insert_line)
    cut_btn.place(anchor=CENTER, x=480, y=665, width=85, height=60)

    cancel = Button(text='Cancel', font=('calibri', 12),bg=button_color, activebackground=color, command=Delete_line)
    cancel.place(anchor=CENTER, x=730, y=665, width=85, height=60)
    main.bind("<Shift_R>", Insert_line)
    main.bind("<space>",Insert_line)

    main.mainloop()