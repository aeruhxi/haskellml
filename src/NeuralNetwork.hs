module NeuralNetwork
  () where

import           Control.Monad         (mapM)
import qualified Data.Sequence         as Seq
import           Debug.Trace
import           Numeric.LinearAlgebra ((<>), (><))
import qualified Numeric.LinearAlgebra as LA

type MatrixNum = LA.Matrix Double
type Cache = ((MatrixNum, MatrixNum, MatrixNum), MatrixNum)

data Parameter = Parameter
  { weightsParam :: MatrixNum
  , biasesParam  :: MatrixNum
  }

-- Utils
emptyx :: MatrixNum
emptyx = (0><0) []

emptyCache :: Cache
emptyCache = ((emptyx, emptyx, emptyx), emptyx)

-- Activation functions
relu :: MatrixNum -> MatrixNum
relu = LA.cmap (max 0)

sigmoid :: MatrixNum -> MatrixNum
sigmoid = LA.cmap (\x -> (1 / (1 + exp (-1 * x))))

relu' :: MatrixNum -> MatrixNum
relu' = LA.cmap f
  where
    f x | x < 0 = 0
        | otherwise = 1

sigmoid' :: MatrixNum -> MatrixNum
sigmoid' xw = (sigmoid xw) * (1 - sigmoid xw)

-- | Initialize parameters. Randomize weights and zeroes biases
initParams :: [Int] -> IO [Parameter]
initParams layerDims =
  (fmap . fmap) (\(bx, wx) -> Parameter wx bx) ((zip biases) <$> weights)
    where
      weights = mapM genParams layerDims'
      layerDims' = (zip layerDims (tail layerDims))
      genParams (prevDim, currentDim) = LA.randn currentDim prevDim
      biases =
        fmap
        (\(_, currentDim) -> (currentDim><1) (replicate currentDim 0))
        layerDims'

-- | Propagate forward to next layer. Calulates next a
linearForward axPrev wx bx activationFun =
  let zx = linearForward' axPrev wx bx
      ax = activationFun zx
      cache = ((axPrev, wx, bx), zx)
      linearForward' axPrev wx bx = (wx <> axPrev) + bx :: MatrixNum
  in  (ax, cache)

-- | Complete forward propagation. Calculates final ax
propagateForward :: MatrixNum
                 -> [Parameter]
                 -> (MatrixNum, [Cache])
propagateForward xx params  =
  (outputAx, hiddenCache ++ [outputCache])
    where
      params' = take ((length params) - 1) params
      f (aPrev, _) (Parameter wx bx) = linearForward aPrev wx bx relu
      hidden = scanl f (xx, emptyCache) params'
      hiddenCache = drop 1 (fmap snd hidden)
      (lastHiddenAx, _) = last hidden
      Parameter wxLast bxLast = last params
      (outputAx, outputCache) = linearForward lastHiddenAx wxLast bxLast sigmoid


-- | Propagate backward to previous layer
linearBackward :: MatrixNum -> Cache -> (MatrixNum -> MatrixNum)
               -> (MatrixNum, MatrixNum, MatrixNum)
linearBackward dax ((axPrev, wx, bx), zx) af' =
  (daxPrev, dwx, dbx)
    where
      dzx = dax * af' zx
      m = fromIntegral $ (snd . LA.size) axPrev
      dwx = (dzx <> (LA.tr axPrev)) / m
      sums = fmap sum (LA.toLists dzx)
      layerSize = (fst . LA.size) wx
      dbx = ((layerSize><1) sums) / m
      daxPrev = (LA.tr wx) <> dzx

-- | Complete backpropagation. Computes gradients (dax, dwx, dbx)
-- of each layer
propagateBackward :: MatrixNum -> (LA.Vector Double) -> [Cache]
                  -> [(MatrixNum, MatrixNum, MatrixNum)]
propagateBackward axOut yx caches =
  let l = length caches
      yx' = LA.reshape 1 yx
      daxOut = - (yx' / axOut) - ((1 - yx') / (1 - axOut))
      gradOut = linearBackward daxOut (last caches) sigmoid'
      hiddenCaches = reverse $ take (length caches - 1) caches
      grads = scanl f gradOut hiddenCaches
      f (dax, _, _) cache = linearBackward dax cache relu'
  in  reverse grads

-- | Perform gradient descent. Computes new decreased gradients (dax, dwx, dbx)
updateParameters :: [Parameter]
                 -> [(MatrixNum, MatrixNum, MatrixNum)]
                 -> Double
                 -> [Parameter]
updateParameters params grads eta = fmap fun (zip params grads)
   where fun ((Parameter wx bx), (_, dwx, dbx)) =
           Parameter
             (wx - (LA.scalar eta) * dwx)
             (bx - (LA.scalar eta) * dbx)

train :: MatrixNum
      -> LA.Vector Double
      -> [Int]
      -> Double
      -> Int
      -> IO [Parameter]
train xx yx layerDims eta iterations =
  do
    params <- initParams layerDims
    return $ train' params iterations
  where
    train' params 0 = params
    train' params times =
      let (ax, caches) = propagateForward xx params
          grads = propagateBackward ax yx caches
          newParams = updateParameters params grads eta
      in  train' newParams (times - 1)
