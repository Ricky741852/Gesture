import datetime
import subprocess
import matplotlib.pyplot as plt
from library import *
import time

plt.rcParams["font.family"] = "Times New Roman"
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def evaluate_on_gemmini(data_folder='gesture_input', model_name='quantized_model_20230918.h'):
    """We run all gesture input files on gemmini using spike simulation and output the results as txt files.
    A large batch size can cause errors in gemmini. This test must be executed in the chipyard environment.
    Remember to run 'source /home/jendy/chipyard/env.sh' in order to activate conda-lock environment.
    """

    program_path = '/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/'
    build_path = '/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/'
    run_path = '/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/'
    output_path = '/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/build_backup/output/'
    target_dir = os.path.join(build_path, 'include', data_folder)
    file_names = os.listdir(target_dir)

    # Run each input header file evaluation on gemmini
    for file_name in file_names:
        if os.path.isfile(os.path.join(target_dir, file_name)):
            print(file_name)

        # gemmini-rocc-tests/include/gesture_top_hfile.h
        top_hfile_path = os.path.join(build_path, 'include', 'gesture_top_hfile.h')
        with open(top_hfile_path, 'w') as file:
            file.write(f'//{datetime.datetime.now()}\n')
            file.write(f'#ifndef GEMMINI_PROJECTS_TOP_HFILE_H\n')
            file.write(f'#define GEMMINI_PROJECTS_TOP_HFILE_H\n')
            file.write(f'#include "include/{model_name}"  // model file\n')
            file.write(f'#include "include/{data_folder}/{file_name}"  // input file\n')
            file.write(f'#endif //GEMMINI_PROJECTS_TOP_HFILE_H\n')

        riscv_file = 'gesture_recognition_on_gemmini'
        save_txt = f'{file_name}.txt'
        command = f'spike --extension=gemmini {riscv_file}-baremetal > {save_txt}'

        os.chdir(program_path)
        if not os.path.exists(program_path + riscv_file + '.c'):
            print("Cannot find the main file...")
        os.chdir(build_path)

        def run_build_command(build_path):
            if os.path.exists(os.path.join(build_path, 'build')):
                subprocess.run(
                    ['sudo rm -r build'],
                    cwd=build_path,
                    shell=True,
                    executable='/bin/bash',
                )

            subprocess.run(
                ['./build.sh'],
                cwd=build_path,
                shell=True,
                executable='/bin/bash',
            )

        def run_spike_command(command, save_txt):
            subprocess.run(
                [command],
                shell=True,
                executable='/bin/bash',
                cwd=run_path,
            )
            path = f'/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/build_backup/output/{save_txt}'

        # Run build compile C code
        run_build_command(build_path)

        # Run spike simulation
        run_spike_command(command, save_txt)
        save_file = os.path.join(run_path, save_txt)
        shutil.copy2(save_file, output_path)

        print(f'Successfully copy {save_file} to {output_path}')


def evaluate_accuracy():
    
    output_path = '/home/jendy/chipyard/generators/gemmini/software/gemmini-rocc-tests/build_backup/output/'
    file_names = os.listdir(output_path)
    gesture_answer, predict_answer = [], []

    # Run each input header file evaluation on gemmini
    for file_name in file_names:
        file_path = os.path.join(output_path, file_name)
    
        with open(file_path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            a, b = last_line.split(',')
            gesture_answer.append(a)
            predict_answer.append(b)

        if a != b:
            print(file_name, a, b)

    plot_confusion_matrix(gesture_answer, predict_answer)


def plot_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix, precision and recall"""

    from sklearn.metrics import confusion_matrix, classification_report

    cm_dir = 'gemmini_c_code'
    check_directories(cm_dir)

    # Generate classification report of precision and recall
    class_list = [
        "Gesture " + str(i) for i in range(1, 6)
    ]  # Gesture 1, ..., Gesture n
    print(Color.H_INFO + f'Classification Report ' + '=' * 60 + Color.RESET)
    print(classification_report(y_true, y_pred, target_names=class_list))

    # Save report to cm_dir/20230426_classification_report.txt
    report = os.path.join(cm_dir, f'{timestamp}_classification_report.txt')
    with open(report, 'w') as f:
        print(
            classification_report(y_true, y_pred, target_names=class_list), file=f
        )

    confusion_mat = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix with values
    plt.imshow(confusion_mat, cmap='Blues')

    # Add text annotations to each cell
    thresh = confusion_mat.max() / 2.0
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black",
                     fontsize=10)

    plt.xticks(range(5), class_list, fontsize=10)
    plt.yticks(range(5), class_list, fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', )
    plt.xlabel('Predicted Gesture', fontsize=12, fontweight='bold')
    plt.ylabel('True Gesture', fontsize=12, fontweight='bold')
    savefig_name = f'{timestamp}_confusion_matrix.png'
    plt.savefig(os.path.join(cm_dir, savefig_name))
    plt.show()


if __name__ == '__main__':

    # Run if you want to spike all file on gemmini
    # evaluate_on_gemmini()

    # Calculate accuracy
    # evaluate_accuracy()

    pass
