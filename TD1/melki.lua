require 'nn'
require 'gnuplot'
require 'tools' 


-- 1: Creation du jeux de données
local DIMENSION=2 -- dimension d'entrée
local n_points=1000 -- nombre de points d'apprentissage

-- Tirage de deux gaussiennes
local mean_positive=torch.Tensor(DIMENSION):fill(1); local var_positive=1.0
local mean_negative=torch.Tensor(DIMENSION):fill(-1); local var_negative=1.0
local xs=torch.Tensor(n_points,DIMENSION)
local ys=torch.Tensor(n_points,1)
for i=1,n_points/2 do  xs[i]:copy(torch.randn(DIMENSION)*var_positive+mean_positive); ys[i][1]=1 end
for i=n_points/2+1,n_points do xs[i]:copy(torch.randn(DIMENSION)*var_negative+mean_negative); ys[i][1]=-1 end

-- 2 : creation du modele
-- TODO
local model = nn.Linear(DIMENSION, 1)
local criterion = nn.MSECriterion()
model:reset(1e-1)


-- 3 : Boucle d'apprentissage
local learning_rate = 1e-2
local maxEpoch = 1000
local all_losses={}
for iteration=1,maxEpoch do
   ------ Mise à jour des paramètres du modèle
   ------ Evaluation de la loss moyenne 
   -- TODO
   model:zeroGradParameters()
   local loss = 0
   ---- calcul de la loss moyenne

   
   -- version gradient stochastique
   for i = 1, n_points do
      local _nidx_ = torch.random(n_points)
      local x, y = xs[_nidx_], ys[_nidx_]
      local output = model:forward(x)
      local loss = criterion:forward(output, y)
      local dloss_do = criterion:backward(output, y)
      model:backward(x, dloss_do)
   end
   


   -- -- version gradient batch
   -- local outputs = model:forward(xs)
   -- local loss = criterion:forward(outputs, ys)
   -- local dloss_do = criterion:backward(outputs, ys)
   -- model:backward(xs, dloss_do)
   
   model:updateParameters(learning_rate)
   all_losses[iteration] = loss  --stockage de la loss moyenne (pour dessin)

   -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
   plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
   --gnuplot.plot(torch.Tensor(all_losses, '~')) 
end



