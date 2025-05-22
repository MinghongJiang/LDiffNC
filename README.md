# LDiffNC
神经崩溃隐空间扩散模型LDiffNC（Latent-based Diffusion Model with Neural Collapse）

$$
L_{LDM}=\mathbb{E} _{t~\left[ 1,T \right] ,y~\left[ 1,K \right] ,z_t,\epsilon _t}\left[ \left\| \epsilon _t-\epsilon _{\theta}\left( z_t,t,y \right) \right\| _{2}^{2} \right] 
$$

$$
L_{Intra}(z_t,y,t)=\mathbb{E} _{t\sim [1,T],y\sim [1,K],z_t}[\parallel f_{\varphi}^{*}(G_{\theta}(z_t,t,y))-\mu _y\parallel _{2}^{2}]
$$

$$
L_{Inter1}(z_t,y,y^{\prime},t)=\mathbb{E} _{t\sim [1,T],y,y^{\prime}\sim [1,K],z_l}[cos(\frac{f_{\varphi}^{*}(G_{\theta}(z_t,t,y))-\alpha \mu _G}{||f_{\varphi}^{*}(G_{\theta}(z_t,t,y))-\alpha \mu _G||_{2}^{2}},\frac{f_{\varphi}^{*}(G_{\theta}(z_t,t,y^{\prime}))-\alpha \mu _G}{||f_{\varphi}^{*}(G_{\theta}(z_t,t,y^{\prime}))-\alpha \mu _G||_{2}^{2}}
$$

$$
L_{Inter2}(z_t,y,y^{\prime},t)=\mathbb{E} _{t\sim [1,T],y,y^{\prime}\sim [1,K],z_t}[\parallel f_{\varphi}^{*}(G_{\theta}(z_t,t,y))-\mu _G\parallel _2-\parallel f_{\varphi}^{*}(G_{\theta}(z_t,t,y^{\prime}))-\mu _G\parallel _2]
$$

$$
L_{Intra}\bigl( z_t,y,t \bigr) =\mathbb{E} _{t\sim [1,T],y\sim [1,K],z_t}[\parallel \bigl( \frac{1}{\sqrt{\bar{\alpha}_t}}z_t-\sqrt{\frac{1}{\bar{\alpha}_t}-1}\boldsymbol{\epsilon }_{\theta} \bigr) -\mu _y\parallel _2]
$$

$$
L_{Inter2}(z_t,y,y^{\prime},t)=\mathbb{E} _{t\sim [1,T],y,y^{\prime}\sim [1,K],z_l}
[\parallel (\frac{1}{\sqrt{\bar{\alpha}_t}}z_{t}^{k}-\sqrt{\frac{1}{\bar{\alpha}_t}-1}\boldsymbol{\epsilon }_{\theta}\bigl) -\mu _G)-((\frac{1}{\sqrt{\bar{\alpha}_t}}z_{t}^{k^{\prime}}-\sqrt{\frac{1}{\bar{\alpha}_t}-1}\boldsymbol{\epsilon }_{\theta} \bigr)
$$

$$
L_{\mathrm{LdiffNC}}=L_{DM}+\beta L_{Intra}+\gamma L_{Inter2}
$$


view master
