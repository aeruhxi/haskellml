module NaiveBayes.GaussianNB
  ( train
  , predict
  , GaussianNB(..)
  ) where
import           Data.List           ((++))
import qualified Data.List           as L
import           Data.Map            ((!))
import qualified Data.Map            as M
import qualified Data.Vector.Unboxed as U
import           Prelude             hiding ((++))
import           Statistics.Matrix   as S
import           Statistics.Sample   as S

-- Implementation taken from
-- https://chrisalbon.com/machine_learning/naive_bayes/naive_bayes_classifier_from_scratch/

-- | Usage Example
--
-- trainFeatures :: S.Matrix
-- trainFeatures = S.fromRowLists
--   [ [6.00, 180, 12]
--   , [5.92, 190, 11]
--   , [5.58, 170, 12]
--   , [5.92, 165, 10]
--   , [5.00, 100, 6]
--   , [5.50, 150, 8]
--   , [5.42, 130, 7]
--   , [5.75, 150, 9]
--   ]
--
-- trainLabels :: U.Vector Int
-- trainLabels = U.fromList [0, 0, 0, 0, 1, 1, 1, 1]
--
-- trainedData = train trainFeatures trainLabels
-- predict trainedData [6.00, 130, 8]
-- 1

data GaussianNB = GaussianNB
  { cMeans      :: M.Map Int [Double]
  , cVariances  :: M.Map Int [Double]
  , cPriorProbs :: M.Map Int Double
  }


  deriving (Show)

labelsCounts :: U.Vector Int -> M.Map Int Int
labelsCounts = U.foldr' (\x acc -> M.insertWith (+) x 1 acc) M.empty

labelsFeatures :: S.Matrix -> U.Vector Int -> M.Map Int [U.Vector Double]
labelsFeatures features labels =
  L.foldr
    (\(l, fs) acc -> M.insertWith (++) l [fs] acc)
    M.empty
    (zip' labels features)

zip' :: U.Vector Int -> S.Matrix -> [(Int, U.Vector Double)]
zip' vec mx = U.ifoldr' (\i x acc -> (x, S.row mx i):acc) [] vec

means :: M.Map Int [U.Vector Double] -> M.Map Int [Double]
means = fmap (fmap S.mean . S.toRows . S.transpose . S.fromRows)

variances :: M.Map Int [U.Vector Double] -> M.Map Int [Double]
variances =
  fmap (fmap S.variance . S.toRows . S.transpose . S.fromRows)

priorProbs :: M.Map Int Int -> Int -> M.Map Int Double
priorProbs mp totalCount =
  fmap (\x -> fromIntegral x / fromIntegral totalCount) mp

train :: Matrix -> U.Vector Int -> GaussianNB
train features labels =
  let countMap = labelsCounts labels
      featuresMap = labelsFeatures features labels
      meansMap = means featuresMap
      variancesMap = variances featuresMap
      priorProbsMap = priorProbs countMap (U.length labels)
  in  GaussianNB meansMap variancesMap priorProbsMap

predict :: GaussianNB -> [Double] -> Int
predict cf feature =
  fst $ L.maximumBy
    (\(_, a) (_, b) -> a `compare` b)
    (M.toList (posteriorProbs cf feature))

posteriorProbs :: GaussianNB -> [Double] -> M.Map Int Double
posteriorProbs (GaussianNB ms vs pp) feature =
  M.mapWithKey (\k x -> x * likelihoods k) pp
    where likelihoods k = L.product $
            L.map
              (\(x, m, v) -> exp ((- ((x - m) ** 2)) / (2 * v)) / (sqrt (2 * pi * v)))
              (L.zip3 feature (ms ! k) (vs ! k))

