import onnx

def main():
    model = onnx.load("../models/lama_dilated-onnx-w8a16/lama_dilated.onnx")
    onnx.save_model(
        model,
        "lama_merged.onnx",
        save_as_external_data=False,
    )

if __name__ == "__main__":
    main()