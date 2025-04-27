import torch

def update_weights(file_path, save_path):
    # Load the model weights
    state_dict = torch.load(file_path, map_location="cpu")
    
    # Update keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("sequence_transititon.3", "sequence_transititon.2")
        new_state_dict[new_key] = value
    
    # Save the updated weights
    torch.save(new_state_dict, save_path)
    print(f"Updated weights saved to {save_path}")

# Example usage
update_weights("../../exps/test39/models/epoch_14/pytorch_model_fsdp.bin", "pl_init.pth")
