function createDataset(cardPoints)
   local cardPoints = cardPoints or 1000

   local pos = torch.Tensor({1,1})
   local neg = torch.Tensor({-1,-1})

   local var_positive=1.0

   local X=torch.Tensor(cardPoints,2)
   local y=torch.Tensor(cardPoints,1)

   for i=1,cardPoints/2 do
      
      X[i]= pos + torch.randn(2)/1.5
      y[i] = 1
   end

   for i=cardPoints/2+1,cardPoints do
      X[i]= neg + torch.randn(2)/1.5
      y[i] = 2
   end

   return X,y
end 
