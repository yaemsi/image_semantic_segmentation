import argparse
import os

from subprocess import call
from loguru import logger

from image_semantic_segmentation import (
    iou,
    custom_compute_metrics
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to benchmark a model on different test sets')
    parser.add_argument('--script_path', type=str, default='./infer.py',
                        help='path of the script to perform inference with the model')
    parser.add_argument('--testset_path', type=str, default='./test/',
                        help='path of the test sets')
    parser.add_argument('--prediction_path', type=str, default='./test/',
                        help='path of the directory where the predictions are saved')
    args = parser.parse_args()

    test_dirs = os.listdir(args.testset_path)
    if not test_dirs:
        test_dirs.append('.')

    score_dict = {}

    script_dir = os.path.dirname(args.script_path)
    logger.info(f"script_dir == {script_dir}")
    script_name = os.path.basename(args.script_path)
    interp_name = './'
    if os.path.splitext(script_name)[1] == '.py':
        interp_name = 'python'
    elif os.path.splitext(script_name)[1] == '.sh':
        interp_name = 'sh'

    for test_name in test_dirs:
        pred_path = os.path.join(args.prediction_path, test_name)
        if not os.path.isdir(pred_path):
            current_wd = os.getcwd()
            os.chdir(script_dir)
            logger.info('Generating the predictions for {} set'.format(test_name))
            call([interp_name, script_name, os.path.join(args.testset_path, test_name, 'img'), pred_path])
            logger.info('Generation done.')
            os.chdir(current_wd)

        test_mask_path = os.path.join(args.testset_path, test_name, 'mask')
        score = custom_compute_metrics(iou, pred_path=pred_path, mask_path=test_mask_path)
        score_dict[test_name] = score

    if len(test_dirs) == 1:
        logger.info('mean IOU on {}: {:.4f}'.format(test_name, score_dict[test_name]))
    else:
        for test_name in test_dirs:
            logger.info('mean IOU on {}: {:.4f}'.format(test_name, score_dict[test_name]))
