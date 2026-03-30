from itertools import chain
from base import BaseModel
from utils.losses import *
from models.decoder import *
from models.encoder import *



class LGTC_CD(BaseModel):
    def __init__(self, num_classes, conf, loss_l=None, len_unsper=None, testing=False, pretrained=True):
        self.num_classes = num_classes
        if not testing:
            assert (loss_l is not None) 

        super(LGTC_CD, self).__init__()
        self.method = conf['method']

        # Supervised losses
        self.loss_l         = loss_l

        # confidence masking (sup mat)
        if self.method != 'supervised':
            self.confidence_thr     = conf['confidence_thr']
            print ('thr: ', self.confidence_thr)

        # Create the model
        self.encoder = LGIEncoder(pretrained=pretrained)

        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.decoder        = Decoder(upscale, decoder_in_ch, num_classes=num_classes)

        self.dimension1 = nn.Conv2d(2048, 64, kernel_size=1, padding=0, bias=False)
        self.dimension2 = nn.Conv2d(2048, 64, kernel_size=1, padding=0, bias=False)

        self.re_decoder1    = ReDecoder(64, 3)
        self.re_decoder2    = ReDecoder(64, 3)

    def forward(self, A_l=None, B_l=None, target_l=None, \
                    WA_ul=None, WB_ul=None, SA_ul=None, SB_ul=None, target_ul=None):
        if not self.training:
            return self.decoder(torch.abs(self.encoder(A_l)[3]-self.encoder(B_l)[3]))
        input_size  = (A_l.size(2), A_l.size(3))

        # If supervised mode only
        if self.method == 'supervised':
            # Supervised loss
            out_l  = self.decoder(torch.abs(self.encoder(A_l)[3]-self.encoder(B_l)[3]))
            loss_l = self.loss_l(out_l, target_l) 
            curr_losses = {'loss_l': loss_l}
            total_loss = loss_l

            if out_l.shape != A_l.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l}

            return total_loss, curr_losses, outs

        # If semi supervised mode
        else:
            # Supervised loss
            out_l  = self.decoder(torch.abs(self.encoder(A_l)[3]-self.encoder(B_l)[3]))
            loss_l = self.loss_l(out_l, target_l) 

            # Get main prediction
            weak_feat_ul = torch.abs(self.encoder(WA_ul)[3] - self.encoder(WB_ul)[3])
            weak_out_ul = self.decoder(weak_feat_ul)
            SA_f1_ul, SA_f2_ul, SA_f3_ul, SA_f4_ul = self.encoder(SA_ul)
            SB_f1_ul, SB_f2_ul, SB_f3_ul, SB_f4_ul = self.encoder(SB_ul)
            strong_feat_ul = torch.abs(SA_f4_ul - SB_f4_ul)
            strong_out_ul = self.decoder(strong_feat_ul)

            # Generate pseudo_label
            weak_prob_ul = F.softmax(weak_out_ul.detach_(), dim=1)
            max_probs, target_ul = torch.max(weak_prob_ul, dim=1)

            # Calculate pixel-level consistency loss
            mask = max_probs.ge(self.confidence_thr).float()
            loss_ul_pc = (F.cross_entropy(strong_out_ul, target_ul, reduction='none') * mask).mean()  # PC_loss

            # Calculate region-level dual-consistency loss
            SA_f4_ul = F.interpolate(self.dimension1(SA_f4_ul), input_size, mode="bilinear", align_corners=False)
            SB_f4_ul = F.interpolate(self.dimension2(SB_f4_ul), input_size, mode="bilinear", align_corners=False)
            loss_ul_rc = region_consistency_loss(SA_f4_ul, SB_f4_ul, target_ul.unsqueeze(1))    # RC_loss

            # Calculate image-level reconstruction loss
            target_mask = 1 - target_ul.unsqueeze(1)
            re4_SA_ul = self.re_decoder1(SA_f4_ul * target_mask)
            re4_SB_ul = self.re_decoder2(SB_f4_ul * target_mask)
            loss_ul_ic_A = F.mse_loss(re4_SA_ul, SA_ul, reduction='mean')
            loss_ul_ic_B = F.mse_loss(re4_SB_ul, SB_ul, reduction='mean')    # IC_loss

            loss_ul = loss_ul_pc + 0.1 * loss_ul_rc + 0.001 * (loss_ul_ic_A + loss_ul_ic_B)

            # record loss
            curr_losses = {'loss_l': loss_l}
            curr_losses['loss_ul'] = loss_ul
            curr_losses['loss_ul_cls'] = loss_ul_pc

            if weak_out_ul.shape != WA_ul.shape:
                out_l = F.interpolate(out_l, size=input_size, mode='bilinear', align_corners=True)
                weak_out_ul = F.interpolate(weak_out_ul, size=input_size, mode='bilinear', align_corners=True)
            outs = {'pred_l': out_l, 'pred_ul': weak_out_ul}

            # Compute the total loss
            total_loss  = loss_l +  loss_ul
            
            return total_loss, curr_losses, outs

    def get_backbone_params(self):
        return self.encoder.parameters()

    def get_other_params(self):
        return chain(self.decoder.parameters(), self.dimension1.parameters(), self.dimension2.parameters(), self.re_decoder1.parameters(), self.re_decoder2.parameters())

