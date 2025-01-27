import torch
from pytorch_lightning import Trainer

if __name__ == '__main__':


    # Step 1: Load both KAD_weights and vit_weights
    kad_path = "/u/home/ijt/Downloads/KAD/VLP-Seminar/KADcheckpoints/KAD_224/best_valid.pt"
    vit_path = "/u/home/ijt/Downloads/KAD/VLP-Seminar/pretrained/vit_base.ckpt"

    kad_weights = torch.load(kad_path, map_location=torch.device('cpu'))
    vit_weights = torch.load(vit_path, map_location=torch.device('cpu'))

    # Extract image and text encoder weights from KAD
    kad_image_weights = kad_weights['image_encoder']
    kad_text_weights = kad_weights['text_encoder']

    # Extract state_dict from ViT weights
    vit_state_dict = vit_weights['state_dict']

    # Step 2: Define mappings between KAD and ViT keys
    # Example mappings (you need to define these based on your specific architecture)
    image_encoder_mappings = {
        # KAD ResNet keys -> ViT keys
        'resnet.conv1.weight': 'img_encoder_q.model.patch_embed.proj.weight',
        'resnet.bn1.weight': 'img_encoder_q.model.blocks.0.norm1.weight',
        'resnet.bn1.bias': 'img_encoder_q.model.blocks.0.norm1.bias',
        # Add more mappings as needed
    }

    text_encoder_mappings = {
        # KAD BERT keys -> ViT keys
        'bert_model.embeddings.word_embeddings.weight': 'img_encoder_q.model.cls_token',
        'bert_model.encoder.layer.0.attention.self.query.weight': 'img_encoder_q.model.blocks.0.attn.qkv.weight',
        # Add more mappings as needed
    }

    # Step 3: Replace values in vit_weights with values from KAD
    replaced_keys = []
    replaced_count = 0  # Counter for successfully replaced keys

    # Replace image encoder weights
    for kad_key, vit_key in image_encoder_mappings.items():
        if kad_key in kad_image_weights and vit_key in vit_state_dict:
            vit_state_dict[vit_key] = kad_image_weights[kad_key]
            replaced_keys.append((kad_key, vit_key))
            replaced_count += 1

    # Replace text encoder weights
    for kad_key, vit_key in text_encoder_mappings.items():
        if kad_key in kad_text_weights and vit_key in vit_state_dict:
            vit_state_dict[vit_key] = kad_text_weights[kad_key]
            replaced_keys.append((kad_key, vit_key))
            replaced_count += 1

    # Step 4: Print replaced keys
    print("Replaced Keys:")
    for kad_key, vit_key in replaced_keys:
        print(f"KAD Key: {kad_key} -> ViT Key: {vit_key}")

    # Print the total number of keys replaced
    print(f"\nTotal number of keys replaced: {replaced_count}")

    # Step 5: Save the modified vit_weights
    #output_path = "/u/home/ijt/Downloads/KAD/VLP-Seminar/modified_vit_weights.ckpt"
    #torch.save(vit_weights, output_path)
    #print(f"\nModified ViT weights saved to: {output_path}")
    

