print("\n" + "="*80)
print("Ablation Study: TCN Only Model")
print("="*80 + "\n")

# Create TCN-only model (replace BiLSTM with linear)
class TCNOnlyModel(nn.Module):
    """Model with TCN but simple projection (no BiLSTM)"""
    def __init__(self, config):
        super(TCNOnlyModel, self).__init__()

        # Encoders
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.visual_encoder = nn.Sequential(*modules)
        if config['model']['visual_encoder']['freeze']:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        self.text_encoder = BertModel.from_pretrained(config['model']['text_encoder']['type'])
        if config['model']['text_encoder']['freeze']:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.tag_embedding = nn.Embedding(
            config['model']['tag_embedding']['vocab_size'],
            config['model']['tag_embedding']['embed_dim']
        )

        # Multimodal dimension
        multimodal_dim = (
            config['model']['visual_encoder']['feature_dim'] +
            config['model']['text_encoder']['feature_dim'] +
            config['model']['tag_embedding']['embed_dim']
        )

        # TCN only (no BiLSTM)
        self.tcn = TemporalConvolutionalNetwork(
            input_size=multimodal_dim,
            num_channels=config['model']['tcn']['num_channels'],
            kernel_size=config['model']['tcn']['kernel_size'],
            dropout=config['model']['tcn']['dropout']
        )

        # Attention and decoders
        self.attention = MultiHeadCrossAttention(
            visual_dim=self.tcn.output_dim,
            text_dim=config['model']['text_encoder']['feature_dim'],
            num_heads=config['model']['attention']['num_heads'],
            dropout=config['model']['attention']['dropout']
        )

        fusion_dim = self.tcn.output_dim + config['model']['text_encoder']['feature_dim']
        self.fusion_proj = nn.Linear(fusion_dim, 512)

        self.image_decoder = ImageDecoder(feature_dim=512, image_size=224)
        self.text_decoder = TextDecoder(
            vocab_size=config['model']['text_decoder']['vocab_size'],
            embed_dim=config['model']['text_decoder']['embed_dim'],
            hidden_dim=config['model']['text_decoder']['hidden_dim'],
            num_layers=2,
            dropout=0.2
        )

    def forward(self, frames, captions, tags):
        # FIXED: Handle captions shape properly
        if captions.dim() == 2:
            batch_size = captions.size(0)
            seq_len = 10
            max_tokens = 20
            captions = captions.view(batch_size, seq_len, max_tokens)
        else:
            batch_size, seq_len, max_tokens = captions.size()

        # Visual encoding
        frames_flat = frames.view(batch_size * seq_len, 3, 224, 224)
        visual_features = self.visual_encoder(frames_flat)
        visual_features = visual_features.view(batch_size * seq_len, -1)
        visual_features = visual_features.view(batch_size, seq_len, -1)

        # Text encoding
        captions_flat = captions.view(batch_size * seq_len, -1)
        with torch.no_grad():
            text_outputs = self.text_encoder(captions_flat, attention_mask=(captions_flat != 0))
        text_features = text_outputs.pooler_output
        text_features = text_features.view(batch_size, seq_len, -1)

        # Tag embedding
        tag_features = self.tag_embedding(tags)
        tag_features = tag_features.mean(dim=2)

        # Multimodal concatenation
        multimodal_features = torch.cat([visual_features, text_features, tag_features], dim=-1)

        # TCN only (no BiLSTM)
        tcn_output = self.tcn(multimodal_features)

        # Attention and fusion
        attended_output, _ = self.attention(tcn_output, text_features)
        fused = torch.cat([attended_output, text_features], dim=-1)
        fused = self.fusion_proj(fused)

        # Decoders - FIXED: Extract last caption properly
        image_features = fused[:, -1, :]
        generated_images = self.image_decoder(image_features)

        last_caption = captions[:, -1, :]  # (batch, max_tokens)
        caption_input = last_caption[:, :-1]  # (batch, max_tokens-1)
        text_logits = self.text_decoder(caption_input)

        return generated_images, text_logits

# Train TCN-only model
tcn_only_model = TCNOnlyModel(config).to(device)
tcn_trainer = Trainer(tcn_only_model, train_loader, val_loader, config, device)

print("Training TCN-only model...")
tcn_trainer.train(max_epochs=5)

# Evaluate
tcn_metrics = evaluate_model(tcn_only_model, test_loader, config, device)
