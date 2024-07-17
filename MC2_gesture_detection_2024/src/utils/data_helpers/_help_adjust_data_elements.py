import os

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        # 如果是-1000行，則保持原樣
        if '-1000' in line:
            processed_lines.append(line.rstrip())
            continue

        # 將每行分割成元素
        elements = line.strip().split(',')
        # 將第一個元素轉換為其兩倍，並進行邊界檢查
        first_element = 2 * int(elements[0])
        first_element = max(-100, min(100, first_element))
        elements[0] = str(first_element)

        #格式化每個元素
        formatted_elements = ['{0:4d}'.format(int(element)) for element in elements]

        # 將處理後的元素重新組合成一行
        processed_line = ','.join(formatted_elements)
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
