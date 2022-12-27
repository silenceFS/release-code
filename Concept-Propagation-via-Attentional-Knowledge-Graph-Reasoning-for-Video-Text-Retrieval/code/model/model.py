import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix

from model.swin_transformer import SwinTransformer
import pickle as pkl
import numpy as np

class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            if arch_config == 'base_patch16_224':
                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                model = SpaceTimeTransformer(num_frames=num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model.state_dict()
                model.load_state_dict(vit_checkpoint, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)


    # def compute_text(self, text_data):
    def compute_text_content(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    # def compute_video(self, video_data):
    def compute_video_content(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention


class FrozenInTime_ACPL(FrozenInTime):
    def __init__(self,
                 video_params,
                 text_params,
                #  projection_dim=512,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__(video_params, text_params, projection_dim=projection_dim, load_checkpoint=load_checkpoint, projection=projection, load_temporal_fix=load_temporal_fix)


        self.text_model_semantic = AutoModel.from_pretrained(text_params['model'])
        self.text_model_semantic.train()

        self.video_model_semantic = SwinTransformer()
        load_pretrained(self.video_model_semantic)
        
        self.txt_proj_semantic = nn.Sequential(nn.ReLU(),nn.Linear(self.text_model.config.hidden_size, projection_dim))

        self.ACPL = 0
        if self.ACPL > 0:
            kg = np.load('KG.npy', allow_pickle=True).item()
            nodes_ft = kg['nodes_ft']
            rel = kg['rel']
            self.nodes_ft = nn.Parameter(torch.Tensor(nodes_ft),requires_grad=False)
            self.rel = nn.Parameter(torch.Tensor(rel),requires_grad=False)
            self.vid_gcns = nn.ModuleList([GraphAttentionLayer(300,32,0.2,0.2) for i in range(self.ACPL)])
            self.vid_proj_semantic = nn.Sequential(nn.Linear(self.rel.shape[0], projection_dim)) 
        else:
            self.vid_proj_semantic = nn.Sequential(nn.Linear(1000, projection_dim)) 
 

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        video_data = data['video']

        text_embeddings_content, text_embeddings_semantic = self.compute_text(text_data)
        video_embeddings_content, video_embeddings_semantic = self.compute_video(video_data)

        return text_embeddings_content, video_embeddings_content, text_embeddings_semantic, video_embeddings_semantic

    def compute_text(self, text_data):
        text_embeddings_content = self.compute_text_content(text_data)
        text_embeddings_semantic = self.compute_text_semantic(text_data)
        return text_embeddings_content, text_embeddings_semantic

    def compute_video(self, video_data):
        video_embeddings_content = self.compute_video_content(video_data)
        video_embeddings_semantic = self.compute_video_semantic(video_data)
        return video_embeddings_content, video_embeddings_semantic

    def compute_text_semantic(self, text_data):
        text_embeddings_semantic = self.text_model_semantic(**text_data).last_hidden_state[:, 0, :]
        text_embeddings_semantic = self.txt_proj_semantic(text_embeddings_semantic)
        return text_embeddings_semantic

    def compute_video_semantic(self, video_data):
        B,K,C,M,N = video_data.shape
        video_data = video_data.view(B*K,C,M,N)
        video_embeddings_semantic = self.video_model_semantic(video_data)
        video_embeddings_semantic = video_embeddings_semantic.view(B,K,-1)
        if self.ACPL>0:
            video_padding = torch.zeros(B,K,self.rel.shape[0]-1000).to(video_embeddings_semantic.device)
            video_embeddings_semantic = torch.cat((video_embeddings_semantic,video_padding),2)
            for gcn in self.vid_gcns:
                video_embeddings_semantic = video_embeddings_semantic.matmul(gcn(self.nodes_ft,self.rel))
        video_embeddings_semantic = self.vid_proj_semantic(video_embeddings_semantic)
        video_embeddings_semantic = torch.mean(video_embeddings_semantic,1)
        return video_embeddings_semantic


def load_pretrained(model):
    model_path = 'swin_small_patch4_window7_224.pth'
    print(f"==============> Loading weight {model_path} for fine-tuning......")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
            # logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            # logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")
            # logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    # logger.warning(msg)
    print(msg)

    print(f"=> loaded successfully '{model_path}'")

    del checkpoint
    torch.cuda.empty_cache()

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    pass
