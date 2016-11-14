require 'nn'
require 'gnuplot'
require 'tools' 
require 'io'

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
local model= nn.Linear(DIMENSION,1)
model:reset(0.1)
local criterion= nn.MSECriterion()


-- 3 : Boucle d'apprentissage
local learning_rate= 1e-2 
local maxEpoch=1000
local all_losses={}

-- local gradType = 'batch'
-- print('Batch Gradient')

ss
local gradType = 'stoch'
print('Stochastic Gradient')


for iteration=1,maxEpoch do
   ------ Mise à jour des paramètres du modèle
   ------ Evaluation de la loss moyenne 
   model:zeroGradParameters()
   local loss=0

   -- version gradient stochastique
   if gradType == 'stoch' then
      for i = 1, n_points do
         local id = torch.random(n_points)
         local x, y = xs[id], ys[id]
         local output = model:forward(x)
         local loss = criterion:forward(output, y)
         local delta = criterion:backward(output, y)
         model:backward(x, delta)
      end
      
      all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)

      
  -- version gradient batch

   elseif gradType == 'batch' then

      local output = model:forward(xs)
      local loss = criterion:forward(output,ys)
      all_losses[iteration]=loss  --stockage de la loss moyenne (pour dessin)

      local delta = criterion:backward(output,ys)
      model:backward(xs,delta)
      
   
   end

   model:updateParameters(learning_rate)
   
   
   -- plot de la frontiere ou plot du loss (utiliser l'un ou l'autre)
   plot(xs,ys,model,100)  -- uniquement si DIMENSION=2
   -- gnuplot.plot(torch.Tensor(all_losses))
end
