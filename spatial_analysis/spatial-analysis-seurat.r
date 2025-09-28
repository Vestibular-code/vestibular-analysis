library(Seurat)
library(patchwork)
library(dplyr)
library(arrow)

# In the upstream analysis, cells are segmented using Cellpose and matched with the matrix, resulting in an Anndata file that contains spatial information in Python.


object = schard::h5ad2seurat_spatial('/your_pathway/aligned_spatial_data.h5ad')# This analysis code is based on the matrix obtained from the upstream matching for analysis.

vln.plot <- VlnPlot(object, features = "nCount_Spatial", pt.size = 0) + theme(axis.text = element_text(size = 4)) + NoLegend()
count.plot <- SpatialFeaturePlot(object, features = "nCount_Spatial",images = "WT") + theme(legend.position = "right")
vln.plot | count.plot

vln.plot <- VlnPlot(object, features = "nFeature_Spatial", pt.size = 0) + theme(axis.text = element_text(size = 4)) + NoLegend()
count.plot <- SpatialFeaturePlot(object, features = "nFeature_Spatial") + theme(legend.position = "right")
vln.plot | count.plot

object <- NormalizeData(object, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
object <- FindVariableFeatures(object, selection.method = "vst", nfeatures = 3000, verbose = FALSE)
object <- ScaleData(object, verbose = FALSE)
object <- RunPCA(object, features = top30$gene, npcs = 100,assay = "Spatial", reduction.name = "pca.Spatial")#Top30 is the marker gene_list from scRNA_seq data
print(object[["pca.Spatial"]], dims = 1:50, nfeatures = 10)

object <- FindNeighbors(object, dims = 1:50,assay = "Spatial", reduction = "pca.Spatial")
object <- FindClusters(object, resolution = 1.5,cluster.name = "seurat_cluster.Spatialed")
object <- RunUMAP(object, dims = 1:50,min.dist = 0.3,verbose = F,n.neighbors = 40,reduction = "pca.Spatial", reduction.name = "umap.Spatial", return.model = T)

DefaultAssay(object) <- "Spatial"
Idents(object) <- "seurat_cluster.Spatialed"
p1 <- DimPlot(object, reduction = "umap.Spatial", label = F) + ggtitle("Spatialed clustering") + theme(legend.position = "bottom")
SpatialDimPlot(object, label = T, repel = T, label.size = 4,image.alpha=0.1)

DefaultAssay(object) <- "Spatial"

markers <- FindAllMarkers(object, assay = "Spatial", only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
markers %>%
    group_by(cluster) %>%
    slice_max(n = 20, order_by = avg_log2FC)
	
# Annotated each cluster
saveRDS（object,file="/your_pathway/aligned_spatial_final_cluster.rds")
