import argparse

from src.models.train import GestureTensorflow
from src.utils.library import *
from src.visualization import Visualization
from src.utils.receive_data import record

MODEL_DIR = 'output/models'  # Directory to save the trained model
DATASETS_DIR = 'data/datasets/testData'  # Directory to save raw data
PROCESSSED_DATASETS_DIR = 'data/processed/datasets'  # Directory to save formatted data 將訓練資料轉換為.pt檔
EVAL_MODEL = 'model_20240717_170440'
WINDOW_SIZE = 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gesture Recognition on Gemmini')

    # Define command-line arguments
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        required=True,
        help='Mode of the program. Choose from: train, test, get_layer, none',
    )
    parser.add_argument(
        '--eval_model',
        type=str,
        default=EVAL_MODEL,
        help=f'Name of the model for evaluation. Default is {EVAL_MODEL}.',
    )
    parser.add_argument(
        '--index',
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
        '--epoch_num',
        type=int,
        default=50,
        help='The number of epochs for training. Default is 50.',
    )
    parser.add_argument(
        '-a_type',
        '--animation_type',
        type=str,
        default='realtime',
        help='The type of animation to run. Default is realtime. Choose from: realtime, sample, simulation.',
    )
    parser.add_argument(
        '-i_type',
        '--image_type',
        type=str,
        default='sample',
        help='The type of image to run. Default is sample. Choose from: sample, rawdata, groundtruth.',
    )


    args = parser.parse_args()

    mode = args.mode #mode
    
    # GestureTensorflow Parameters
    epoch_num = args.epoch_num

    # Test Mode Parameters
    eval_model = args.eval_model
    index = args.index
    test_all = args.test_all
    plot_cm = args.plot_cm

    # Visualization Parameters
    animation_type = args.animation_type
    image_type = args.image_type

    print(Color.INFO + f'Now running mode: {mode}' + Color.RESET)

    run = GestureTensorflow(
        model_path=MODEL_DIR,
        save_path=PROCESSSED_DATASETS_DIR,
        class_num=4,
        windows_size=WINDOW_SIZE,
        epoch_num=epoch_num,
    )
    

    if mode == 'train':
        run.build_model()
        run.train_model()

    elif mode == 'test':
        run.test_model(
            model_evaluation=eval_model,
            index=index,
            test_all=test_all,
            plot_cm=plot_cm
        )

    elif mode == 'get_layer':
        run.get_model_layer_weights(model_evaluation=eval_model)

    elif mode == 'animation':
        print(Color.MSG + f'Animating {animation_type}...' + Color.RESET)
        visualization = Visualization(EVAL_MODEL, DATASETS_DIR, WINDOW_SIZE)
        visualization.animation(animation_type)
    
    elif mode == 'image':
        print(Color.MSG + f'Visualizing {image_type}...' + Color.RESET)
        visualization = Visualization(EVAL_MODEL, DATASETS_DIR, WINDOW_SIZE)
        visualization.image(image_type)

    elif mode == 'record':
        record()

    elif mode == 'none':
        pass

    else:
        raise ValueError(Color.RED + f'\nInvalid mode value: {mode}' + Color.RESET)

    pass
