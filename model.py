from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import torch
torch.set_printoptions(threshold=100000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
    )
    
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)
        
class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        self.Conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels = 1,out_channels = 1,kernel_size = 3,padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            )
        self.se_1 = SELayer(channel=64, reduction=16)

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            )
        self.Conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels = 1,out_channels = 1,kernel_size = 3,padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            )
        self.se_2 = SELayer(channel=128, reduction=16)

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            )     
        self.Conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels = 1,out_channels = 1,kernel_size = 3,padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            )
        self.se_3 = SELayer(channel=256, reduction=16)

        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            )     
        self.Conv3d_4 = nn.Sequential(
            nn.Conv3d(in_channels = 1,out_channels = 1,kernel_size = 3,padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            )
        self.se_4 = SELayer(channel=512, reduction=16)

        self.pooling4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )    

        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            )    
        self.Conv3d_5 = nn.Sequential(
            nn.Conv3d(in_channels = 1,out_channels = 1,kernel_size = 3,padding = 1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            )
        self.se_5 = SELayer(channel=512, reduction=16)

        self.stage6_7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            ) 


    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7

        """

        out = self.stage1(image) # 2X
        out = self.Conv3d_1(out.unsqueeze(dim=1)).squeeze(dim=1)
        out = self.se_1(out)
        out = self.stage2(out) # 4X
        out = self.Conv3d_2(out.unsqueeze(dim=1)).squeeze(dim=1)
        out = self.se_2(out)
        out = self.stage3(out) # 8X
        out = self.Conv3d_3(out.unsqueeze(dim=1)).squeeze(dim=1)
        out = self.se_3(out)
        
        conv4_3_feats = self.stage4(out)
        out = self.pooling4(conv4_3_feats)   # 16X
        out = self.Conv3d_4(out.unsqueeze(dim=1)).squeeze(dim=1) 
        out = self.se_4(out)
        out = self.stage5(out)  # 16X
        out = self.Conv3d_5(out.unsqueeze(dim=1)).squeeze(dim=1)
        out = self.se_5(out)
        conv7_feats = self.stage6_7(out) # 16X
        
        return conv4_3_feats, conv7_feats


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.stage8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            ) 
        self.stage9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            ) 
        self.stage10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
            ) 
        self.stage11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
            ) 


    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """

        conv8_2_feats = self.stage8(conv7_feats) # 10*10
        conv9_2_feats = self.stage9(conv8_2_feats) # 5*5
        conv10_2_feats = self.stage10(conv9_2_feats) # 3*3
        conv11_2_feats = self.stage11(conv10_2_feats) # 1*1

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes, priors_cxcy):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes
        self.priors_cxcy = priors_cxcy

        # Definir el número de cajas por capa
        n_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}

        # Convoluciones para las cajas de ubicación
        self.loc_convs = nn.ModuleDict({
            k: nn.Conv2d(v[0], n_boxes[k] * 4, kernel_size=3, padding=1)
            for k, v in zip(n_boxes.keys(), [(512, 38), (1024, 19), (512, 10), (256, 5), (256, 3), (256, 1)])
        })

        # Convoluciones para las predicciones de clases
        self.cl_convs = nn.ModuleDict({
            k: nn.Conv2d(v[0], n_boxes[k] * n_classes, kernel_size=3, padding=1)
            for k, v in zip(n_boxes.keys(), [(512, 38), (1024, 19), (512, 10), (256, 5), (256, 3), (256, 1)])
        })

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        # Predicciones de las cajas de ubicación y clases
        locs = []
        classes_scores = []

        for (name, conv_feat) in zip(['conv4_3', 'conv7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2'],
                                     [conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats]):
            locs.append(self.loc_convs[name](conv_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
            classes_scores.append(self.cl_convs[name](conv_feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes))

        locs = torch.cat(locs, dim=1)
        classes_scores = torch.cat(classes_scores, dim=1)

        #print("Final locs shape:", locs.shape)
        #print("Final classes_scores shape:", classes_scores.shape)
        #print("Expected priors size:", self.priors_cxcy.size(0))

        # Asegúrate de que el número de priors coincida con el número de predicciones
        if self.priors_cxcy.size(0) != locs.size(1):
            locs = locs[:, :self.priors_cxcy.size(0), :]
            classes_scores = classes_scores[:, :self.priors_cxcy.size(0), :]

        assert self.priors_cxcy.size(0) == locs.size(1) == classes_scores.size(1), \
            "Mismatch in number of priors and size of predicted locations and scores"

        return locs, classes_scores



class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()  # Asegúrate de que esta función esté generando 8732 cajas
        self.pred_convs = PredictionConvolutions(n_classes, self.priors_cxcy)

    def forward(self, image):
        conv4_3_feats, conv7_feats = self.base(image)  
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  
        conv4_3_feats = conv4_3_feats / norm  
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)  

        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        aspect_ratios = {'conv4_3': [1., 2., 0.5], 
                        'conv7': [1., 2., 3., 0.5, .333], 
                        'conv8_2': [1., 2., 3., 0.5, .333],
                        'conv9_2': [1., 2., 3., 0.5, .333],
                        'conv10_2': [1., 2., 0.5],
                        'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())  # Lista de feature maps
        prior_boxes = []

        # Revisa el número de cajas generadas por cada capa
        total_boxes = 0

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                        total_boxes += 1  # Contar las cajas generadas

                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
                            total_boxes += 1

        # Revisa el total de cajas generadas
        print(f"Total prior boxes generated: {total_boxes}")

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)
        return prior_boxes






class MultiBoxLoss(nn.Module): 
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy # torch.Size([8732, 4])
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)  
        n_classes = predicted_scores.size(2)


        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0) 

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  

            _, prior_for_each_object = overlap.max(dim=1)  
            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)，这个解决了1
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1. 

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  


            # Store
            true_classes[i] = label_for_each_prior 

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  


        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  
        n_hard_negatives = self.neg_pos_ratio * n_positives  


        # First, find the loss for all priors
        #conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  
        conf_loss_all = self.cross_entropy(predicted_scores.reshape(-1, n_classes), true_classes.reshape(-1))

        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)，
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss



# test
if __name__ == '__main__':
    
    import torchsummary
    from thop import clever_format,profile
    model = SSD300(n_classes=20).cuda()
    # torchsummary.summary(model,(1,96,300,300))
    # print(sum(param.numel() for param in model.parameters()))

    inp = torch.rand(1,96,300,300).cuda()
    macs, params = profile(model, inputs=(inp, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)