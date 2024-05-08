import os

def process_file(file_path):
    # 讀取文件的所有行
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 過濾掉包含 '-1000,-1000,-1000,-1000,-1000' 的行
    lines = [line for line in lines if '-1000,-1000,-1000,-1000,-1000' not in line]

    # 將修改後的行寫回到文件中
    with open(file_path, 'w') as file:
        file.writelines(lines)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == "__main__":
    # 指定主資料夾路徑
    main_folder_path = 'E:\Gesture\Jendy_Code\gesture_recognition_on_gemmini_6gesture\\otherData'  # 請替換為你的實際路徑

    # 執行處理
    process_folder(main_folder_path)
