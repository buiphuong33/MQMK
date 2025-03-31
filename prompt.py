import torch
import torch.nn as nn
import math
class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,multi_query=False,multi_key=False,class_per_task=20,k_key=1,class_group=1):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.multi_query = multi_query
        self.multi_key = multi_key
        self.class_per_task = class_per_task
        self.k_key = k_key
        self.class_group = class_group
        self.group_key_num = math.ceil(self.class_per_task/self.class_group)

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size=10, length=20, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            if self.multi_key:
                if self.class_group > 1:
                    assert self.class_group<=self.class_per_task
                    key_shape = (pool_size, self.group_key_num, embed_dim)
                else:
                    key_shape = (pool_size, class_per_task, embed_dim)
            else:
                key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, query=False,task_id =None,target=None,fast3=False):
        if fast3:
            self.multi_query = False
        out = dict()
        # print('e_prompt_forward')
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            # print(prompt_key_norm.shape, x_embed_norm.shape)
            if query==False:
                if self.multi_query==False:
                    # print('single query!')
                    if self.multi_key==False:
                        similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
                        similarity = similarity.t() # B, pool_size
                        (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                    else:
                        prompt_key_norm_copy = prompt_key_norm.unsqueeze(0)
                        B = x_embed_norm.shape[0]
                        P = prompt_key_norm_copy.shape[1]
                        C = x_embed_norm.shape[1]
                        prompt_key_norm_copy = prompt_key_norm_copy.expand(B, P,self.class_per_task, C)
                        x_embed_norm_copy = x_embed_norm
                        x_embed_norm_copy = x_embed_norm_copy.unsqueeze(1)
                        x_embed_norm_copy = x_embed_norm_copy.unsqueeze(1)
                        x_embed_norm_copy = x_embed_norm_copy.expand(B, P, self.class_per_task, C)
                        similarity = torch.sum(prompt_key_norm_copy*x_embed_norm_copy, dim=-1)
                        # print(similarity.shape)
                        if target is None:
                            similarity = torch.topk(similarity,dim=-1,k=self.k_key)[0]
                            similarity = torch.sum(similarity,dim=-1)
                        else:
                            class_index = target % self.class_per_task
                            similarity = similarity[torch.arange(B).unsqueeze(1), :, class_index.unsqueeze(1)].squeeze(1)
                        (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=-1)
                    out['similarity'] = similarity
                    # print(similarity.shape)
                else:
                    if self.multi_key==False:
                    # print('multi query!')
                    # print(cls_features)
                        prompt_key_norm_copy = prompt_key_norm.unsqueeze(0)
                        B = x_embed_norm.shape[0]
                        P = prompt_key_norm_copy.shape[1]
                        C = x_embed_norm.shape[2]
                        prompt_key_norm_copy = prompt_key_norm_copy.expand(B, P, C)

                        # q_all = q_all.unsqueeze(2)
                        # q_all = q_all.expand(B, N, task_class_num, C)
                        similarity = torch.sum(prompt_key_norm_copy*x_embed_norm, dim=-1)
                        # similarity = torch.topk(similarity,dim=-1,k=1)[0]
                        # similarity = torch.sum(similarity,dim=-1)
                        # print(similarity.shape)
                        # print(similarity.shape)
                        (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=-1) # B, top_k
                        out['similarity'] = similarity
                    else:
                        # multi query and multi key

                        ## use group key or not
                        if self.class_group >= 1:
                            prompt_key_norm_copy = prompt_key_norm.unsqueeze(0)
                            B = x_embed_norm.shape[0]
                            P = prompt_key_norm_copy.shape[1]
                            C = x_embed_norm.shape[2]
                            prompt_key_norm_copy = prompt_key_norm_copy.expand(B, P, self.group_key_num,C)
                            x_embed_norm_copy = x_embed_norm.unsqueeze(2)
                            x_embed_norm_copy = x_embed_norm_copy.expand(B, P, self.group_key_num, C)

                            similarity = torch.sum(prompt_key_norm_copy*x_embed_norm_copy, dim=-1)

                            if target is None:
                                similarity = torch.topk(similarity,dim=-1,k=self.k_key)[0]
                                similarity = torch.sum(similarity,dim=-1)
                            else:
                                group_index = (target % self.class_per_task) // self.class_group
                                similarity = similarity[torch.arange(B).unsqueeze(1), :, group_index.unsqueeze(1)].squeeze(1)
                            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=-1)
                            out['similarity'] = similarity
                        
                        else:
                            prompt_key_norm_copy = prompt_key_norm.unsqueeze(0)
                            B = x_embed_norm.shape[0]
                            P = prompt_key_norm_copy.shape[1]
                            C = x_embed_norm.shape[2]
                            prompt_key_norm_copy = prompt_key_norm_copy.expand(B, P, self.class_per_task,C)
                            x_embed_norm_copy = x_embed_norm.unsqueeze(2)
                            x_embed_norm_copy = x_embed_norm_copy.expand(B, P, self.class_per_task, C)

                            # q_all = q_all.unsqueeze(2)
                            # q_all = q_all.expand(B, N, task_class_num, C)
                            similarity = torch.sum(prompt_key_norm_copy*x_embed_norm_copy, dim=-1)
                            # print(similarity.shape)
                            if target is None:
                            
                                # print('-------------------------')
                                # print('k_key',self.k_key)
                                # print('-------------------------')
                                
                                similarity = torch.topk(similarity,dim=-1,k=self.k_key)[0]
                                # print(similarity.shape)
                                similarity = torch.sum(similarity,dim=-1)
                            else:
                                # print(similarity[:2])
                                # print(target[:2])
                                class_index = target % self.class_per_task
                                # class_index: [B]
                                # torch.arange(B): [0,1,2,3,4,5,6,7,8,9,...,B]
                                # [B,1]
                                # similarity: [B, L, class_per_task]
                                
                                similarity = similarity[torch.arange(B).unsqueeze(1), :, class_index.unsqueeze(1)].squeeze(1)
                                # print(similarity[:2])
                                # print(similarity.shape)
                                # print(similarity.shape)
                                # print(similarity.shape)
                            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=-1) # B, top_k
                            out['similarity'] = similarity

                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
            
            if prompt_mask is not None:
                idx = prompt_mask # B, top_k
            
            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:,:,idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                # batched_prompt_raw = batched_prompt_raw.permute(0, 2, 1, 3, 4, 5, 6)
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:,idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                # batched_prompt_raw = batched_prompt_raw.permute(0, 2, 1, 3, 4, 5, 6)
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )
            # print(prompt_key_norm.shape)
            if self.multi_key:
                batched_key_norm = prompt_key_norm[idx]
            else:
                batched_key_norm = prompt_key_norm[idx] # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            if query:
                # x_embed_norm = x_embed_norm.unsqueeze(1)
                # print(batched_key_norm.shape, x_embed_norm.shape)

                # sim = batched_key_norm * x_embed_norm
                # # print(batched_key_norm.shape, x_embed_norm.shape)
                reduce_sim = 0
            else:
                # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
                # print(batched_key_norm.shape, x_embed_norm.shape)
                # sim = batched_key_norm * x_embed_norm # B, top_k, C
                sim = similarity
                if prompt_mask is not None:
                    sim = sim[:,task_id]
                else:
                    # print('task_id',task_id)
                    sim[:,task_id+1:]=-1
                # if self.multi_key==False and self.multi_query==False:
                #     sim[:,task_id+1:]=-1
                reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar


            # out['reduce_sim'] = 0
            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt

        if fast3:
            self.multi_query = True
        return out