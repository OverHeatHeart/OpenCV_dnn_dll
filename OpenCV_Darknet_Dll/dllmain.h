#pragma once
extern "C" int __declspec(dllexport) TestFrame();
vector<String> getOutputsNames(const Net& net);
void postprocess(Mat& frame, const vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
