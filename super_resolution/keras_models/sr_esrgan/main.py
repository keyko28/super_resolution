from train import train

if __name__ == '__main__':

    p1 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_LR_unknown\\X4'
    p2 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_train_HR'
    p3 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_valid_LR_unknown\\X4'
    p4 = 'D:\\pet_projects\\super_resolution\\.div2k\\images\\DIV2K_valid_HR'

    path = 'D:\\pet_projects\\sr_esrgan\\Losses'

    train(100, 'df', model_save_dir = path, paths = [p1, p2, p3, p4, 'png'])