//-------------Registering transformations and use it on atlas labels------------

// ============ SIMPLE ELASTIX ============
//----------------------- Rigid and affine -------------------------

var elastix = new SimpleElastix();
elastix.LogToFileOff();

elastix.SetOutputDirectory(output_rigidaffine);

elastix.SetParameterMap(new VectorOfParameterMap()
{
	elastix.ReadParameterFile(param_rigid),
	elastix.ReadParameterFile(param_affine)
});

var brainatlas_image = SimpleITK.ReadImage(brainatlas);

_baselineImage = SimpleITK.Cast(_baselineImage, PixelIDValueEnum.sitkInt16);
brainatlas_image = SimpleITK.Cast(brainatlas_image, PixelIDValueEnum.sitkInt16);

elastix.SetFixedImage(_baselineImage);
elastix.SetMovingImage(brainatlas_image);

try
{
	elastix.Execute();
}
catch (Exception e)
{
	Console.Out.WriteLine(e);
	throw;
}
var rigidTransformParameterResult = elastix.GetTransformParameterMap();
var rigidAffineResult = elastix.GetResultImage();

rigidTransformParameterResult[0]["ResultImagePixelType"] = new VectorString() { "int" };
rigidTransformParameterResult[0]["FixedInternalImagePixelType"] = new VectorString() { "int" };
rigidTransformParameterResult[0]["MovingInternalImagePixelType"] = new VectorString() { "int" };

rigidTransformParameterResult[1]["ResultImagePixelType"] = new VectorString() { "int" };
rigidTransformParameterResult[1]["FixedInternalImagePixelType"] = new VectorString() { "int" };
rigidTransformParameterResult[1]["MovingInternalImagePixelType"] = new VectorString() { "int" };

elastix.WriteParameterFile(rigidTransformParameterResult[0],
	output_rigidaffine + "\\TransformParameters.0.txt");
elastix.WriteParameterFile(rigidTransformParameterResult[1],
	output_rigidaffine + "\\TransformParameters.1.txt");

Console.Out.WriteLine("Rigid-affine registration done. ");

////----------------------- B-spline -------------------------

elastix = new SimpleElastix();
elastix.LogToFileOff();

elastix.SetOutputDirectory(output_bspline);

elastix.SetParameterMap(new VectorOfParameterMap()
{
	elastix.ReadParameterFile(param_bspline)
});

//var brainmask_image = SimpleITK.ReadImage(brainmask);
//brainmask_image = SimpleITK.Cast(brainmask_image, PixelIDValueEnum.sitkUInt8);

elastix.SetFixedImage(_baselineImage);
elastix.SetMovingImage(rigidAffineResult);
//elastix.SetFixedMask(brainmask_image);

try
{
	elastix.Execute();
}
catch (Exception e)
{
	Console.Out.WriteLine(e);
	throw;
}
var bsplineTransformParameterResult = elastix.GetTransformParameterMap();
var bsplineResult = elastix.GetResultImage();


bsplineTransformParameterResult[0]["ResultImagePixelType"] = new VectorString() { "int" };
bsplineTransformParameterResult[0]["FixedInternalImagePixelType"] = new VectorString() { "int" };
bsplineTransformParameterResult[0]["MovingInternalImagePixelType"] = new VectorString() { "int" };

elastix.WriteParameterFile(bsplineTransformParameterResult[0],
	output_bspline + "\\TransformParameters.0.txt");

Console.Out.WriteLine("Bspline registration done. ");

// --------------------------------- Transform labels ------------------------------
var atlas_image = SimpleITK.ReadImage(originalAtlas);
atlas_image = SimpleITK.Cast(atlas_image, PixelIDValueEnum.sitkInt16);

var transformix = new SimpleTransformix();
transformix.LogToFileOff();
transformix.SetOutputDirectory(Path.GetDirectoryName(transformedAtlas));

transformix.SetMovingImage(atlas_image);
transformix.SetTransformParameterMap(
	rigidTransformParameterResult
);

try
{
	transformix.Execute();
}
catch (Exception e)
{
	Console.Out.WriteLine(e);
	throw;
}

var rigidAffineTransformed = transformix.GetResultImage();
rigidAffineTransformed = SimpleITK.Cast(rigidAffineTransformed, PixelIDValueEnum.sitkInt16);

var writer = new ImageFileWriter();
writer.SetFileName(output_rigidaffine + "\\rigidaffine_result.mhd");
writer.Execute(rigidAffineTransformed);

transformix = new SimpleTransformix();
transformix.LogToFileOff();
transformix.SetOutputDirectory(Path.GetDirectoryName(transformedAtlas));

transformix.SetMovingImage(rigidAffineTransformed);
transformix.SetTransformParameterMap(
	bsplineTransformParameterResult
);

try
{
	transformix.Execute();
}
catch (Exception e)
{
	Console.Out.WriteLine(e);
	throw;
}
_atlasImage = transformix.GetResultImage();
Console.Out.WriteLine("Transformation of atlas done. ");

_atlasImage = SimpleITK.Cast(_atlasImage, PixelIDValueEnum.sitkInt16);

// Write output image
writer.SetFileName(transformedAtlas);
writer.Execute(_atlasImage);
Console.Out.WriteLine("Registered atlas written to file.");