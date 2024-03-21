from fastapi import FastAPI, File, UploadFile, HTTPException
from train import Net, train, test, main
import torch
from torchvision import transforms
import io

app = FastAPI()

# Define global variables
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.post("/run_inference/")
async def run_inference(image: UploadFile = File(...)):
    # Load image
    contents = await image.read()
    img = transforms.ToTensor()(Image.open(io.BytesIO(contents))).unsqueeze(0)

    # Load model checkpoint
    checkpoint = torch.load("best_model_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return {"prediction": prediction}


@app.post("/run_train/")
async def run_train(config_file: UploadFile = File(...)):
    try:
        # Save configuration file
        contents = await config_file.read()
        with open("new_config.yaml", "wb") as f:
            f.write(contents)

        # Run training with new configuration file
        main(["--config", "new_config.yaml"])

        return {"message": "Model training completed with new configuration."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))