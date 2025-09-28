import java.io.*
import java.nio.file.*
import java.util.*
import bigwarp.landmarks.*
import bigwarp.transforms.*
import net.imglib2.realtransform.*
import net.imglib2.realtransform.inverse.*

// 定义样本名列表
def sam_name_list = [
    "wt1",
    "wt2",
    "wt3",
    "wt4",
    "wt5",
    "wt6",
    "wt7",
    "wt8",
    "tmie1",
    "tmie2",
    "tmie3",
    "tmie4",
    "tmie5",
    "tmie6"
]
def rootDir = "D:/Ruiqi/vg_space_mapping/wt/data/"

// 其他参数
def inverseOrForward = "Moving to target"  // 或 "Target to moving"
def transformType = "Thin Plate Spline"   // 或 "Affine", "Similarity", "Rotation", "Translation"
def invTolerance = 0.1
def invMaxIters = 200
def csvHasHeader = false
def sx = 1.0
def sy = 1.0
def sz = 1.0
def needInverseTransform = true // 定义变量

def buildTransform(File landmarksPath, String transformType, int nd, boolean needInverse, double invTolerance, int maxIters) {
    // load the transform
    def ltm = new LandmarkTableModel(nd)
    try {
        ltm.load(landmarksPath)
    } catch (IOException e) {
        e.printStackTrace()
        return
    }

    def bwTransform = new BigWarpTransform(ltm, transformType)
    def xfm = bwTransform.getTransformation()

    if (xfm instanceof Wrapped2DTransformAs3D) {
        xfm = ((Wrapped2DTransformAs3D) xfm).getTransform()
    }

    if (needInverse) {
        if (transformType.equals("Thin Plate Spline")) {
            xfm.getOptimzer().setMaxIters(maxIters)
            xfm.getOptimzer().setTolerance(invTolerance)
        }
        xfm = xfm.inverse()
    }

    return xfm
}

// 遍历每个样本名
sam_name_list.each { sam_name ->
    println("Processing sample: $sam_name")

    // 根据样本名生成根目录
    def rootDir1 = rootDir + "$sam_name/six_point/"
    println("Root directory: $rootDir1")

    // 根据根目录生成具体的路径
    def landmarksPath = new File(rootDir1 + "usethis_mapping_landmarks.csv")
    def inCsv = new File(rootDir1 + "test_a.csv")
    def outCsv = new File(rootDir1 + "testoutput.csv")

    // 检查文件是否存在
    if (!landmarksPath.exists() || !inCsv.exists()) {
        println("Skipping sample $sam_name: Missing landmarks or input CSV file.")
        return
    }

    // read the input points
    List<String> lines
    try {
        lines = Files.readAllLines(Paths.get(inCsv.getAbsolutePath()))
    } catch (IOException e) {
        e.printStackTrace()
        return null
    }

    // get the transformation to apply
    int nd = lines.get(0).split(",").length
    def transform = buildTransform(landmarksPath, transformType, nd, needInverseTransform, invTolerance, invMaxIters)

    // transform all points
    def outputLines = []
    def result = new double[nd]
    boolean firstLine = true
    for (def l : lines) {
        // add the first line to the output if it's a header
        if (firstLine && csvHasHeader) {
            outputLines.add(l)
            firstLine = false
            continue
        }

        // parse line
        def pt = l.split(",").collect { s -> Double.parseDouble(s) }
        def scale = [sx, sy, sz] as double[]

        // elementwise multiplication of pt and scale
        def scaledpt = [pt, scale].transpose().collect { it[0] * it[1] }

        // transform point
        try {
            transform.apply(scaledpt as double[], result)
        } catch (Exception e) {
            System.err.println("Warning: failed to transform " + Arrays.toString(scaledpt))
            Arrays.fill(result, Double.NaN)
        }

        outputLines.add(result.collect { x -> Double.toString(x) }.join(","))
    }

    // write output
    try {
        Files.write(Paths.get(outCsv.getAbsolutePath()), outputLines)
    } catch (IOException e) {
        e.printStackTrace()
    }
    println("Processing complete for sample: $sam_name")
}

println("All samples processed!")