import torch
import torchvision

class _preprocess:
    def __init__(self):
        super().__init__()
        ...
    def __call__(self, batch, **kwargs):
        kwargs # dummy line
        ... # TODO add pre-process if required
        number_of_images, channels, height, width = batch.shape
        number_of_images, channels, height, width # FIXME dummy line
        kernel_size = (8, 8)
        pad_height, pad_width = torch.Tensor([height, width]) % torch.Tensor(kernel_size)
        pad_height, pad_width = int(pad_height), int(pad_width)
        batch = torch.nn.Unfold(kernel_size = kernel_size, 
                                dilation = (1, 1), 
                                padding = (pad_height, pad_width),
                                stride = (8, 8),
                                )(batch)
        batch = batch.view(-1, 3, 8, 8)
        return batch, (pad_height, pad_width)

class _postprocess:
    def __init__(self):
        super().__init__()
        ...
    def __call__(self, keypoints, descriptors, padding, **kwargs):
        kwargs # dummy line
        ... # TODO add post-process if required
        pad_height, pad_width = padding
        keypoints = keypoints.view(10, 1*8*8, -1)
        descriptors = descriptors.view(10, 64*8*8, -1)
        keypoints = torch.nn.Fold(
                                output_size = (100, 100), 
                                kernel_size = (8, 8), 
                                dilation = (1, 1), 
                                padding = (pad_height, pad_width), 
                                stride = (8, 8),
                                )(keypoints)
        descriptors = torch.nn.Fold(
                                output_size = (100, 100), 
                                kernel_size = (8, 8), 
                                dilation = (1, 1), 
                                padding = (pad_height, pad_width), 
                                stride = (8, 8),
                                )(descriptors)
        return keypoints, descriptors

class _feature_encoder(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.div_1 = torch.nn.Sequential(
                            # torch.nn.AvgPool2d( # number of parameters = 0
                            #                   kernel_size=(2, 2),
                            #                   ),
                            torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                           in_channels  = 3,
                                           out_channels = 32,
                                           kernel_size  = (3, 3),
                                           stride       = (1, 1),
                                           padding      = (1, 1),
                                           bias         = True,
                                           ),
                            torch.nn.BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                                num_features = 32, # previous `out_channels`
                                                ),
                            torch.nn.ReLU( # number of parameters = 0
                                         inplace = True,
                                         ),
                            ).to(self.device)
        self.div_2 = torch.nn.Sequential(
                                        torch.nn.AvgPool2d( # number of parameters = 0
                                                          kernel_size=(2, 2),
                                                          count_include_pad = False,
                                                          ),
                                        torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                                       in_channels  = 3,
                                                       out_channels = 32,
                                                       kernel_size  = (3, 3),
                                                       stride       = (1, 1),
                                                       padding      = (1, 1),
                                                       bias         = True,
                                                       ),
                                        torch.nn.BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                                            num_features = 32, # previous `out_channels`
                                                            ),
                                        torch.nn.ReLU( # number of parameters = 0
                                                     inplace = True,
                                                     ),
                                        ).to(self.device)
        ...
    def forward(self, batch):
        ... # TODO extract multi-scale features
        output = []
        output.append(self.div_1(batch))
        output.append(self.div_2(batch))
        
        features = tuple(feature for feature in output)
        return features
    
class _feature_aggregator(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.div_1 = torch.nn.Sequential(
                                        torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                                       in_channels  = 32,
                                                       out_channels = 1,
                                                       kernel_size  = (1, 1),
                                                       stride       = (1, 1),
                                                       padding      = (0, 0),
                                                       bias         = True,
                                                       ),
                                        torch.nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True),
                                        ).to(self.device)
        self.div_2 = torch.nn.Sequential(
                                        torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                                       in_channels  = 32,
                                                       out_channels = 1,
                                                       kernel_size  = (1, 1),
                                                       stride       = (1, 1),
                                                       padding      = (0, 0),
                                                       bias         = True,
                                                       ),
                                        torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        ).to(self.device)
        self.upsample = (
                        self.div_1, 
                        self.div_2
                        )
        ...
    def forward(self, features):
        ... # TODO aggregate multi-scale features
        features = torch.cat(list(upsample(feature) for upsample, feature in zip(self.upsample, features)), dim=1)
        return features
    
class _keypoint_decoder(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.deformable_offset = torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                                in_channels  = 2,
                                                out_channels = 2 * 5 * 5,
                                                kernel_size  = (3, 3),
                                                stride       = (1, 1),
                                                padding      = (1, 1),
                                                bias         = True,
                                                ).to(self.device)
        self.deformable_conv = torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                              in_channels  = 2,
                                              out_channels = 1,
                                              kernel_size  = (5, 5),
                                              stride       = (1, 1),
                                              padding      = (2, 2),
                                              bias         = True,
                                              ).to(self.device)
        self.normalize = torch.nn.BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                             num_features = 1, # previous `out_channels`
                                             ).to(self.device)
        self.softmax = torch.nn.Softmax2d()
    def forward(self, features):
        ... # TODO extract keypointness features
        deformable_offset = self.deformable_offset(
                                                  features
                                                  )
        scores_map = torchvision.ops.deform_conv2d(
                                                 input   = features,
                                                 offset  = deformable_offset,
                                                 weight  = self.deformable_conv.weight,
                                                 bias    = self.deformable_conv.bias,
                                                 padding = (2, 2),
                                                 mask    = None
                                                 )
        scores_map = self.normalize(scores_map)
        scores_map = self.softmax(scores_map)
        return scores_map
    
class _descriptor_decoder(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.deformable_offset = torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                                in_channels  = 2,
                                                out_channels = 2 * 5 * 5,
                                                kernel_size  = (3, 3),
                                                stride       = (1, 1),
                                                padding      = (1, 1),
                                                bias         = True,
                                                ).to(self.device)
        self.deformable_conv = torch.nn.Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                              in_channels  = 2,
                                              out_channels = 64,
                                              kernel_size  = (5, 5),
                                              stride       = (1, 1),
                                              padding      = (2, 2),
                                              bias         = True,
                                              ).to(self.device)
        self.normalize = torch.nn.BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                             num_features = 64, # previous `out_channels`
                                             ).to(self.device)
        ...
    def forward(self, features):
        ... # TODO extract descriptor features
        deformable_offset = self.deformable_offset(
                                                  features
                                                  )
        score_map = torchvision.ops.deform_conv2d(
                                                 input   = features,
                                                 offset  = deformable_offset,
                                                 weight  = self.deformable_conv.weight,
                                                 bias    = self.deformable_conv.bias,
                                                 padding = (2, 2),
                                                 mask    = None
                                                 )
        descriptors = self.normalize(score_map)
        return descriptors
    
class Network(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess = _preprocess()
        self.backbone = torch.nn.Sequential(
            _feature_encoder(),
            _feature_aggregator(),
            )
        self.head_keypoint = _keypoint_decoder()
        self.head_descriptor = _descriptor_decoder()
        self.postprocess = _postprocess()
        ...
    def forward(self, batch):
        ...
        batch = batch.to(self.device)
        batch, padding = self.preprocess(batch)
        features = self.backbone(batch)
        keypoints = self.head_keypoint(features)
        descriptors = self.head_descriptor(features)
        keypoints, descriptors = self.postprocess(keypoints, descriptors, padding)
        # keypoints: here is the keypoint score
        # needs one step post process to select the points with max score in a window and abose specific threshold
        return keypoints, descriptors
    
if __name__ == "__main__":
    print(f"Nerual Network Demo")
    dummy_input = torch.randint(0, 255, (10, 3, 100, 100), dtype=torch.float)
    dummy_output = torch.randint(0, 255, (10, 100, 100), dtype=torch.float)
    model = Network()
    print(f"{model.__class__.__name__} has {sum(p.numel() for p in model.parameters())} parameters")
    
    print(f"input: {dummy_input.shape}")
    keypoints, descriptors = model(dummy_input)
    print(f"keypoints: {keypoints.shape}")
    print(f"descriptors: {descriptors.shape}")
    
    print(f"*"*80)
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser(
                prog="Basketball Analysis Train Demo",
                description='Basketball Analysis Train Demo Description',
                epilog='Basketball Analysis Train Demo Footer',
                )
    arg = parser.parse_args()
    arg.device = None
    arg.log_every_epoch = 10
    arg.save_every_epoch = 1
    arg.validate_every_epoch = 10
    arg.dataset_reallocate_every_epoch = 10
    arg.epochs = 1000
    print(f"*"*80)
    print(f"Provided Args...")
    print(f"{'device':20s}: {arg.device}")
    print(f"{'log_every_epoch':20s}: {arg.log_every_epoch}")
    print(f"{'save_every_epoch':20s}: {arg.save_every_epoch}")
    print(f"{'validate_every_epoch':20s}: {arg.validate_every_epoch}")
    print(f"{'dataset_reallocate_every_epoch':20s}: {arg.dataset_reallocate_every_epoch}")
    print(f"{'epochs':20s}: {arg.epochs}")
    print(f"*"*80)
    
    torch.random.manual_seed(42)
    
    _device = arg.device # force specific device for training
    device = _device if _device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model = Network().to(device)
    
    def loss_fn(output, target):
        target = target.to(output) # update target dtype and device if required
        return torch.mean(torch.pow(output-target, 2), dtype=torch.float)
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print(f"*"*80)
    print(f"Train Started Using Device `{device}`")
    for epoch in range(1, arg.epochs+1):
        logging_msg = []
        logging_msg.append(f"Epoch {epoch: 6d}")
        epoch_train_loss = 0
        dummy_input = torch.randint(0, 255, (10, 3, 100, 100), dtype=torch.float)
        dummy_output = torch.randn((10, 100, 100), dtype=torch.float)
        image, keypoints_truth = dummy_input, dummy_output
        # Sets model to TRAIN mode
        model.train()
        # Forward pass: Compute predicted y by passing x to the model
        keypoints_pred, descriptors_pred = model(image.type(torch.float))
        keypoints_pred = keypoints_pred.view(-1, 100, 100)
        # Compute loss
        loss = loss_fn(keypoints_pred, keypoints_truth.to(keypoints_pred))
        logging_msg.append(f"loss {loss: 6.2f}")
        epoch_train_loss += loss
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(", ".join(logging_msg))
        import cv2
        print(f"dummy_input: {dummy_input.shape}")
        print(f"dummy_output: {dummy_output.shape}")
        print(f"keypoints_pred: {keypoints_pred.shape}")
        dummy_input = dummy_input[0].cpu().numpy()
        dummy_output = cv2.cvtColor(dummy_output[0].cpu().numpy(), cv2.COLOR_GRAY2BGR)
        keypoints_pred = cv2.cvtColor(keypoints_pred[0].detach().cpu().numpy(), cv2.COLOR_GRAY2BGR)
        
        print(f"dummy_input: {dummy_input.shape}")
        print(f"dummy_output: {dummy_output.shape}")
        print(f"keypoints_pred: {keypoints_pred.shape}")
        dummy_input = dummy_input.transpose(1, 2, 0)
        print(f"dummy_input: {dummy_input.shape}")
        print(f"dummy_output: {dummy_output.shape}")
        print(f"keypoints_pred: {keypoints_pred.shape}")

        im = cv2.hconcat([dummy_input, dummy_output, keypoints_pred])
        cv2.imshow("", im)
        cv2.waitKey(1)