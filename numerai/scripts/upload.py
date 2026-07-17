import os
import argparse
from numerapi import NumerAPI
from dotenv import load_dotenv  # Add this import

def main():
    # Add this line to automatically load variables from your .env file
    load_dotenv() 

    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-path", required=True,
                        help="Path to the cloudpickled predict() file")
    parser.add_argument("--model-name", required=True,
                        help="Name of the model as registered on numer.ai")
    args = parser.parse_args()

    # These will now successfully pull from your .env file
    public_id = os.environ["NUMERAI_PUBLIC_ID"]
    secret_key = os.environ["NUMERAI_SECRET_KEY"]

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)

    models = napi.get_models()
    if args.model_name not in models:
        raise ValueError(
            f"Model '{args.model_name}' not found. "
            f"Available: {list(models.keys())}. "
            f"Register it first on numer.ai (Models tab -> Create Model)."
        )
    model_id = models[args.model_name]

    print(f"Uploading {args.pkl_path} to model '{args.model_name}' ({model_id})...")
    upload_id = napi.model_upload(args.pkl_path, model_id=model_id)
    print(f"Upload submitted. Upload ID: {upload_id}")
    print("Check status on numer.ai under your model's Submissions tab.")
    print("Statuses cycle: Pending -> Downloading -> Running -> Succeeded/Failed")

if __name__ == "__main__":
    main()