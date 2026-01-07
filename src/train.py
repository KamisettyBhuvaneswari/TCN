class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler - FIXED: removed 'verbose' parameter
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['training']['lr_scheduler']['patience'],
            factor=config['training']['lr_scheduler']['factor'],
            min_lr=config['training']['lr_scheduler']['min_lr']
        )

        # Loss function
        self.loss_fn = StorytellingLoss()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_image_loss = 0
        total_text_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            frames = batch['frames'].to(self.device)
            captions = batch['captions'].to(self.device)
            tags = batch['tags'].to(self.device)
            target_frame = batch['target_frame'].to(self.device)
            target_caption = batch['target_caption'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            generated_images, text_logits = self.model(frames, captions, tags)

            # Loss computation
            loss, img_loss, txt_loss = self.loss_fn(
                generated_images, target_frame,
                text_logits, target_caption
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_norm']
            )
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_image_loss += img_loss.item()
            total_text_loss += txt_loss.item()

            pbar.set_postfix({
                'loss': loss.item(),
                'img_loss': img_loss.item(),
                'txt_loss': txt_loss.item()
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_img_loss = total_image_loss / len(self.train_loader)
        avg_txt_loss = total_text_loss / len(self.train_loader)

        self.train_losses.append(avg_loss)

        return avg_loss, avg_img_loss, avg_txt_loss

    def val_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                captions = batch['captions'].to(self.device)
                tags = batch['tags'].to(self.device)
                target_frame = batch['target_frame'].to(self.device)
                target_caption = batch['target_caption'].to(self.device)

                generated_images, text_logits = self.model(frames, captions, tags)

                loss, _, _ = self.loss_fn(
                    generated_images, target_frame,
                    text_logits, target_caption
                )

                total_loss += loss.item()
                pbar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def train(self, max_epochs):
        """Train model for multiple epochs"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")

        for epoch in range(1, max_epochs + 1):
            # Train epoch
            train_loss, train_img_loss, train_txt_loss = self.train_epoch(epoch)

            # Validation epoch
            val_loss = self.val_epoch(epoch)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{max_epochs}")
            print(f"  Train Loss: {train_loss:.6f} (Image: {train_img_loss:.6f}, Text: {train_txt_loss:.6f})")
            print(f"  Val Loss: {val_loss:.6f}")

            # Get current learning rate and print it
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.2e}")

            # Model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save checkpoint
                checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config['training']['early_stopping_patience']}")

                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    print("\n✓ Early stopping triggered!")
                    break

        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)

# Initialize trainer
trainer = Trainer(model, train_loader, val_loader, config, device)

print("✓ Trainer initialized!")
