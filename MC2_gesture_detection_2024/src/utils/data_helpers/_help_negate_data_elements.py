import os

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        # 將每個元素轉換為相反數
        elements = [str(-1 * int(element)) for element in line.strip().split(',')]
        processed_line = ', '.join(elements)
        processed_lines.append(processed_line)

    with open(file_path, 'w') as file:
        file.write('\n'.join(processed_lines))

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == "__main__":
    # 指定主資料夾路徑
    main_folder_path = 'E:\Gesture\Jendy_Code\gesture_recognition_on_gemmini_6gesture\\trainData'  # 請替換為你的實際路徑

    # 執行處理
    process_folder(main_folder_path)
