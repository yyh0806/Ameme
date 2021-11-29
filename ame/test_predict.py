import ame.dataset.test
import matplotlib.pyplot as plt

from trainer import Trainer
from metric import *
from loguru import logger
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn import *
from tqdm import tqdm

from ame.utils import *
from ame.dataset.dataloaders import *

from config import *


def post_process(probability, threshold=0.5, min_size=300):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = []
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            a_prediction = np.zeros((520, 704), np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions


def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def check_is_run_length(mask_rle):
    if not mask_rle:
        return True
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    start_prev = starts[0]
    ok = True
    for start in starts[1:]:
        ok = ok and start > start_prev
        start_prev = start
        if not ok:
            return False
    return True


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiments/cell.yml")
    cfg.freeze()
    logger.info(cfg)
    # seed
    fix_all_seeds(cfg['SEED'])
    # dataloader
    test_path = os.path.join(cfg["DATALOADER"]["ARGS"]["data_path"], 'test')
    ds_test = ame.dataset.test.TestCellDataset(test_path)
    dl_test = DataLoader(ds_test, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    # build model architecture
    model = None
    # --------------------------------- segmentation ---------------------------------
    encoder = None
    decoder = None
    segmentation_head = None
    classification_head = None
    if cfg["MODEL"]["TYPE"] == "segmentation":
        from segmentation.encoders import Encoder
        from segmentation.decoders import Decoder
        from segmentation.model import SegModel
        from segmentation.base import SegmentationHead, ClassificationHead

        encoder = Encoder(**cfg["MODEL"]["ARGS"]["Encoder"])
        decoder = Decoder(**cfg["MODEL"]["ARGS"]["Decoder"])
        segmentation_head = SegmentationHead(**cfg["MODEL"]["ARGS"]["Segmentation_head"])
        if "Classification_head" in cfg["MODEL"]["ARGS"]:
            classification_head = ClassificationHead(**cfg["MODEL"]["ARGS"]["Classification_head"])
        assert encoder is not None
        assert decoder is not None
        assert segmentation_head is not None
        model = SegModel(encoder, decoder, segmentation_head, classification_head)
    # --------------------------------------------------------------------------------
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['N_GPU'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(cfg['TEST_CHECKPOINT'], map_location=device)['state_dict'])
    model.eval()
    submission = []
    for i, batch in enumerate(tqdm(dl_test)):
        preds = model(batch['image'].to(device))
        preds = preds.detach().cpu().numpy()[:, 0, :, :]  # (batch_size, 1, size, size) -> (batch_size, size, size)
        for image_id, probability_mask in zip(batch['id'], preds):
            try:
                # if probability_mask.shape != IMAGE_RESIZE:
                #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
                predictions = post_process(probability_mask)
                for prediction in predictions:
                    try:
                        submission.append((image_id, rle_encoding(prediction)))
                    except:
                        print("Error in RL encoding")
            except Exception as e:
                print(f"Exception for img: {image_id}: {e}")

            # Fill images with no predictions
            image_ids = [image_id for image_id, preds in submission]
            if image_id not in image_ids:
                submission.append((image_id, ""))

    df_submission = pd.DataFrame(submission, columns=['id', 'predicted'])
    df_submission.to_csv(cfg['OUTPUT'], index=False)

    if df_submission['predicted'].apply(check_is_run_length).mean() != 1:
        df = pd.DataFrame([(f[:-4], "") for f in os.listdir(test_path)], columns=['id', 'predicted'])
        df.to_csv(cfg['OUTPUT'], index=False)
