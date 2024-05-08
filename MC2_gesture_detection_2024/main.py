import argparse

from train_model import *
from library import *

MODEL_DIR = 'saved_model_h5'  # Directory to save the trained model
SAVE_DIR = 'format_data'  # Directory to save formatted data
INCLUDE_DIR = 'include_header'  # Directory for include headers
SOURCE_DIR = 'gemmini_c_code'  # Directory for Gemmini C code
GEMMINI_PATH = os.path.join(
    os.path.expanduser('~'),
    'chipyard',
    'generators',
    'gemmini',
    'software',
    'gemmini-rocc-tests',
)  # Path to Gemmini
CFILE_NAME = 'gesture_recognition_on_gemmini.c'  # Name of the C source file
# EVAL_MODEL = 'model_20230426_150158.h5'  # Default model for evaluation
EVAL_MODEL = 'model_20240318_184740.h5'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gesture Recognition on Gemmini')

    # Define command-line arguments
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        required=True,
        help='Mode of the program. Choose from: train, test, write_model, write_input, get_threshold, get_layer, none',
    )
    parser.add_argument(
        '--eval_model',
        type=str,
        default=EVAL_MODEL,
        help=f'Name of the model for evaluation. Default is {EVAL_MODEL}.',
    )
    parser.add_argument(
        '--single_idx',
        type=int,
        default=0,
        help='Index of the single file to test. Default is 0.',
    )
    parser.add_argument(
        '--test_all',
        action='store_true',
        help='Whether to test all files. Default is False.',
    )
    parser.add_argument(
        '--plot_cm',
        action='store_true',
        help='Whether to plot confusion matrix during testing. Default is False.',
    )
    parser.add_argument(
        '--post_process',
        action='store_true',
        help='Whether to do post-process during testing. Default is False.',
    )
    parser.add_argument(
        '--windows_stride',
        type=int,
        default=20,
        help='The stride of sliding window. Default is 20.',
    )
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=50,
        help='The number of epochs for training. Default is 50.',
    )

    args = parser.parse_args()

    mode = args.mode #mode
    
    # GestureTensorflow Parameters
    windows_stride = args.windows_stride
    epoch_num = args.epoch_num

    # Test Mode Parameters
    eval_model = args.eval_model
    single_idx = args.single_idx
    test_all = args.test_all
    plot_cm = args.plot_cm
    post_process = args.post_process

    print(Color.INFO + f'Now running mode: {mode}' + Color.RESET)

    check_directories(MODEL_DIR, SAVE_DIR, INCLUDE_DIR, SOURCE_DIR)

    run = GestureTensorflow(
        model_path=MODEL_DIR,
        save_path=SAVE_DIR,
        include_path=INCLUDE_DIR,
        class_num=6,
        windows_size=50,
        windows_stride=windows_stride,
        epoch_num=epoch_num,
    )

    if mode == 'train':
        run.build_model()
        run.train_model()

    elif mode == 'test':
        run.test_model(
            model_evaluation=eval_model,
            single_idx=single_idx,
            test_all=test_all,
            plot_cm=plot_cm,
            post_process=post_process
        )

    elif mode == 'write_model':
        quantized_model_name = run.quantize_model(model_quantization=eval_model)
        copy_hfile_to_gemmini(quantized_model_name, INCLUDE_DIR, GEMMINI_PATH)

    elif mode == 'write_input':
        quantized_input_dir = run.quantize_input(single_idx=single_idx, write_all=True)
        copy_hdir_to_gemmini(quantized_input_dir, GEMMINI_PATH, folder='gesture_input')

    elif mode == 'get_threshold':
        run.predict_zero_label_gesture(model_evaluation=eval_model, do_eval=True)

    elif mode == 'get_layer':
        run.get_model_layer_weights(model_evaluation=eval_model)

    elif mode == 'none':
        pass

    else:
        raise ValueError(Color.RED + f'\nInvalid mode value: {mode}' + Color.RESET)

    # copy_cfile_for_git(CFILE_NAME, GEMMINI_PATH, SOURCE_DIR)
    # copy_hfile_for_git('func.h', GEMMINI_PATH, INCLUDE_DIR)

    pass
