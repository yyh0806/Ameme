
class CellDataset(Dataset):
    def __init__(self, df: pd.core.frame.DataFrame, train: bool):
        self.IMAGE_RESIZE = (448, 448)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)
        self.df = df
        self.base_path = TRAIN_PATH
        self.gb = self.df.groupby('id')
        self.transforms = Compose([Resize(self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]),
                                   Normalize(mean=self.RESNET_MEAN, std=self.RESNET_STD, p=1),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5)])

        # Split train and val set
        all_image_ids = np.array(df_train.id.unique())
        np.random.seed(42)
        iperm = np.random.permutation(len(all_image_ids))
        num_train_samples = int(len(all_image_ids) * 0.9)

        if train:
            self.image_ids = all_image_ids[iperm[:num_train_samples]]
        else:
            self.image_ids = all_image_ids[iperm[num_train_samples:]]

    def __getitem__(self, idx: int) -> dict:

        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)

        # Read image
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)

        # Create the mask
        mask = build_masks(df_train, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # print(np.moveaxis(image,0,2).shape)
        return np.moveaxis(np.array(image), 2, 0), mask.reshape((1, self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)
