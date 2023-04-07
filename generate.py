import torch
from model import Transformer
from tokenizer import process_timeframe, frame2midi, process_noteframe
import layers

def load_model(filepath: str) -> Transformer:
    file = torch.load(filepath)
    model = Transformer(**file["hyper_params"]).to("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.load_state_dict(file["model_state_dict"])
    model.eval()
    return model.to("cuda:0" if torch.cuda.is_available() else "cpu")


def generate(model: object, x: list, max_length: int=1000, min_length: int=100) -> torch.Tensor:
    if len(x) == 0:
        x = [0]
    if x[0] != 1:
        x = [1] + x
    x = torch.tensor(x).to("cuda:0" if torch.cuda.is_available() else "cpu").long()
    x = x.unsqueeze(0)
    count_zero = 0
    with torch.no_grad():
        while True:
            if max_length == 0:
                break
            pred = model(x)
            k_top_pred = torch.topk(pred[0, -1, :], k=10, dim=0)
            pred_idx = torch.distributions.Categorical(logits=k_top_pred.values).sample()

            pred = k_top_pred.indices[pred_idx]
            if pred.item() == 1:
                if x.size(1) >= min_length:
                    break
                continue
                # break
            max_length -= 1
            if pred.item() == 0:
                count_zero += 1
                if count_zero == min_length:
                    break
            x = torch.cat((x, pred.view(1, 1)), dim=1)
    return x.squeeze()[1:]

if __name__ == "__main__":
    model_note = load_model("./checkpoints/model_note_frames_v3.pt")
    model_time = load_model("./checkpoints/model_time_frames_v3.pt")

    token = [1]

    seg_note = generate(model=model_note, x=token, min_length=40, max_length=200)
    seg_time = generate(model=model_time, x=token, min_length=100, max_length=200)
    
    time_frame = process_timeframe(seg_time)
    note_frame = process_noteframe(seg_note)

    print(time_frame)
    print(note_frame)

    torch.save(time_frame, "./data_gen/pred_time_frame_v3.pt")
    torch.save(note_frame, "./data_gen/pred_note_frame_v3.pt")

    midi_obj = frame2midi(note_frame, time_frame, save_path="./midi_gen/b3_2_4.mid", numerator=2)