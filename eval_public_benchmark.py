from test import main_test


if __name__ == '__main__':
    main_test(
        json_path='options/test_seg_paper.json',
        model_path='weights/your_trained_model.pth',
        save_suffix='seg_eval'
    )
