library(dplyr)
library(Seurat)
library(qs)
library(sctransform)
library(utils)
library(stats)

combined<-readRDS(paste0("/path_to_your_result/", "clustered.rds"))
object<-subset(combined,idents="neuron")
object = subset(object, subset = nFeature_RNA > 200 & nCount_RNA > 3000 & percent.mt < 5)

# Regress
DefaultAssay(object) <- "RNA"
object.list <- SplitObject(object, split.by = "batch")
sex.gene = c("Xist", "Ddx3x", "Uty") #"Eif2x3y")
mix.gene = c("Mpz", "Mbp", "Mt1", "Mt2","Pmp22","Igfbp6","Apoe","Dcn","Chgb","Actb","S100a6")
object[["percent.mix"]] <- PercentageFeatureSet(object = object, features = mix.gene, assay = 'RNA')
object[["percent.sex"]] <- PercentageFeatureSet(object = object, features = sex.gene, assay = 'RNA')
mt.gene <- rownames(object)[grep("^mt-", rownames(object))]
object[["percent.mt"]] <- PercentageFeatureSet(object = object, features = mt.gene, assay = 'RNA')
rbc.gene <- rownames(object)[grep("^Hb[ab]-", rownames(object))]
object[["percent.rbc"]] <- PercentageFeatureSet(object = object, features = rbc.gene, assay = 'RNA')
.vars.to.regress = c("percent.mt","percent.rbc","percent.mix")
object.list <- lapply(X = object.list, FUN = function(x) {
  x <- SCTransform(x, method = "glmGamPoi", vars.to.regress = .vars.to.regress, verbose = FALSE)
  })
# object.list<- SCTransform(object.list, method = "glmGamPoi", vars.to.regress = .vars.to.regress, verbose = FALSE)

#? SCTransform(object, method = "glmGamPoi", vars.to.regress = .vars.to.regress, verbose = FALSE)
features <- SelectIntegrationFeatures(object.list = object.list, nfeatures = 3000)
VariableGenes <- setdiff(setdiff(
  head(features, 3000),
  mt.gene), sex.gene)
object.list <- PrepSCTIntegration(object.list = object.list, anchor.features = VariableGenes)

object.anchors <- FindIntegrationAnchors(object.list = object.list, normalization.method = "SCT", anchor.features = VariableGenes)
object<- IntegrateData(anchorset = object.anchors, normalization.method = "SCT")
object<- RunPCA(object, verbose = FALSE)

mt.gene <- rownames(object)[grep("^mt-", rownames(object))]
VariableGenes <- setdiff(setdiff(
  head(VariableFeatures(object), 3000),
  mt.gene), sex.gene)

DefaultAssay(object) <- "integrated"
object
object<- RunPCA(object, features = VariableGenes,npcs=50)
print(object[["pca"]], dims = 1:50, nfeatures = 10)

ndims = 33
object<- FindNeighbors(object, dims = 1:ndims,k.param = 30,do.plot = F)
object<- FindClusters(object, resolution = 0.15)
object<- RunUMAP(object, dims = 1:ndims, min.dist = 0.2,n.neighbors =40)

DefaultAssay(object) <- "SCT"

.test.use = "wilcox"
object.markers <- FindAllMarkers(object, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, test.use = .test.use, return.thresh = 0.01)
object.markers %>%
    group_by(cluster) %>%
    slice_max(n = 20, order_by = avg_log2FC)


saveRDS(object,file=paste0("/path_to_your_result/", "clustered_neuron.rds"))
