/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

#include "LAppView.hpp"
#include <math.h>
#include <string>
#include "LAppPal.hpp"
#include "LAppDelegate.hpp"
#include "LAppLive2DManager.hpp"
#include "LAppTextureManager.hpp"
#include "LAppDefine.hpp"
#include "LAppSprite.hpp"
#include "LAppModel.hpp"

#include <Rendering/OpenGL/CubismOffscreenSurface_OpenGLES2.hpp>
#include <Rendering/OpenGL/CubismRenderer_OpenGLES2.hpp>

#include "JniBridgeC.hpp"

using namespace std;
using namespace LAppDefine;

LAppView::LAppView():
    _programId(0),
    _changeModel(false),
//    _nextModelPath("Hiyori/"),
//    _nextModelJsonFileName("Hiyori.model3.json"),
    _modelParameters(),
    _renderSprite(NULL),
    _renderTarget(SelectTarget_None)
{
    _clearColor[0] = 1.0f;
    _clearColor[1] = 1.0f;
    _clearColor[2] = 1.0f;
    _clearColor[3] = 0.0f;

    // デバイス座標からスクリーン座標に変換するための
    _deviceToScreen = new CubismMatrix44();

    // 画面の表示の拡大縮小や移動の変換を行う行列
    _viewMatrix = new CubismViewMatrix();
}

LAppView::~LAppView()
{
    _renderBuffer.DestroyOffscreenSurface();
    delete _renderSprite;

    delete _viewMatrix;
    delete _deviceToScreen;
}

void LAppView::Initialize()
{
    int width = LAppDelegate::GetInstance()->GetWindowWidth();
    int height = LAppDelegate::GetInstance()->GetWindowHeight();

    float ratio = static_cast<float>(height) / static_cast<float>(width);
    float left = ViewLogicalLeft;
    float right = ViewLogicalRight;
    float bottom = -ratio;
    float top = ratio;

    _viewMatrix->SetScreenRect(left, right, bottom, top); // デバイスに対応する画面の範囲。 Xの左端, Xの右端, Yの下端, Yの上端

    float screenW = fabsf(left - right);
    _deviceToScreen->LoadIdentity();
    _deviceToScreen->ScaleRelative(screenW / width, -screenW / width);
    _deviceToScreen->TranslateRelative(-width * 0.5f, -height * 0.5f);

    // 表示範囲の設定
    _viewMatrix->SetMaxScale(ViewMaxScale); // 限界拡大率
    _viewMatrix->SetMinScale(ViewMinScale); // 限界縮小率

    // 表示できる最大範囲
    _viewMatrix->SetMaxScreenRect(
        ViewLogicalMaxLeft,
        ViewLogicalMaxRight,
        ViewLogicalMaxBottom,
        ViewLogicalMaxTop
    );
}

void LAppView::InitializeShader()
{
    _programId = LAppDelegate::GetInstance()->CreateShader();
}

void LAppView::Resize(float scale)
{
    _modelParameters.modelScale = scale;
}

void LAppView::TranslateX(float x)
{
    _modelParameters.modelTranslateX = x;
}

void LAppView::TranslateY(float y)
{
    _modelParameters.modelTranslateY = y;
}

void LAppView::TranslateFace(float x, float y, float z)
{
    _modelParameters.faceX = x;
    _modelParameters.faceY = y;
    _modelParameters.faceZ = z;
}

void LAppView::Breath(bool enabled)
{
    _modelParameters.breathEnabled = enabled;
}

void LAppView::AutoBlinkEyes(bool enabled)
{
    _modelParameters.autoBlinkEyesEnabled = enabled;
}

void LAppView::ModelEyeLOpen(float value)
{
    _modelParameters.eyeLOpen = value;
}

void LAppView::ModelEyeROpen(float value)
{
    _modelParameters.eyeROpen = value;
}

void LAppView::ModelMouthForm(float value)
{
    _modelParameters.mouthForm = value;
}

void LAppView::ModelMouthOpenY(float value)
{
    _modelParameters.mouthOpenY = value;
}

void LAppView::ModelEyeBallX(float value)
{
    _modelParameters.eyeBallX = value;
}

void LAppView::ModelEyeBallY(float value)
{
    _modelParameters.eyeBallY = value;
}

void LAppView::ModelManualEyeBall(float value)
{
    _modelParameters.manualEyeBallEnabled = value;
}

void LAppView::Render()
{
    if(_changeModel)
    {
        _changeModel = false;
        LAppLive2DManager::GetInstance()->ChangeModelTo(_nextModelPath, _nextModelJsonFileName);
        // LAppLive2DManager::GetInstance()->NextScene();
    }

    LAppLive2DManager* Live2DManager = LAppLive2DManager::GetInstance();
    // Cubism更新・描画
    Live2DManager->OnUpdate(_modelParameters);
    _modelParameters.changeMotion = false;
    _modelParameters.stopMotion = false;
    _modelParameters.resetMotion = false;
    _modelParameters.changeExpression = false;
    _modelParameters.stopExpression = false;

    GLenum glErr = glGetError();
    if (glErr != GL_NO_ERROR)
    {
        LAppPal::PrintLog("[APP]bili L2D glGetError (UPDATE): %d", glGetError());
    }

    // 各モデルが持つ描画ターゲットをテクスチャとする場合
    if (_renderTarget == SelectTarget_ModelFrameBuffer && _renderSprite)
    {
        const GLfloat uvVertex[] =
        {
            1.0f, 1.0f,
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
        };

        for (csmUint32 i = 0; i < Live2DManager->GetModelNum(); i++)
        {
            float alpha = GetSpriteAlpha(i); // サンプルとしてαに適当な差をつける
            _renderSprite->SetColor(1.0f, 1.0f, 1.0f, alpha);

            LAppModel *model = Live2DManager->GetModel(i);
            if (model)
            {
                _renderSprite->RenderImmidiate(model->GetRenderBuffer().GetColorBuffer(), uvVertex);
            }
        }
    }
}

void LAppView::ChangeModelTo(std::string modelPath, std::string modelJsonFileName)
{
    _nextModelPath = modelPath;
    _nextModelJsonFileName = modelJsonFileName;
    _changeModel = true;
}

void LAppView::ApplyMotion(std::string motionGroup, Csm::csmInt32 index)
{
    _modelParameters.nextMotionGroup = motionGroup;
    _modelParameters.nextMotionIndex = index;
    _modelParameters.changeMotion = true;
}

void LAppView::StopMotion()
{
    _modelParameters.stopMotion = true;
}

void LAppView::ResetMotion()
{
    _modelParameters.stopMotion = true;
    _modelParameters.resetMotion = true;
}

void LAppView::ApplyExpression(const char* expressionName) {
    _modelParameters.changeExpression = true;
    _modelParameters.nextExpressionName = expressionName;
}


void LAppView::StopExpression() {
    _modelParameters.stopExpression = true;
}

float LAppView::TransformViewX(float deviceX) const
{
    float screenX = _deviceToScreen->TransformX(deviceX); // 論理座標変換した座標を取得。
    return _viewMatrix->InvertTransformX(screenX); // 拡大、縮小、移動後の値。
}

float LAppView::TransformViewY(float deviceY) const
{
    float screenY = _deviceToScreen->TransformY(deviceY); // 論理座標変換した座標を取得。
    return _viewMatrix->InvertTransformY(screenY); // 拡大、縮小、移動後の値。
}

float LAppView::TransformScreenX(float deviceX) const
{
    return _deviceToScreen->TransformX(deviceX);
}

float LAppView::TransformScreenY(float deviceY) const
{
    return _deviceToScreen->TransformY(deviceY);
}

void LAppView::PreModelDraw(LAppModel &refModel)
{
    // 別のレンダリングターゲットへ向けて描画する場合の使用するフレームバッファ
    Csm::Rendering::CubismOffscreenSurface_OpenGLES2* useTarget = NULL;

    if (_renderTarget != SelectTarget_None)
    {// 別のレンダリングターゲットへ向けて描画する場合

        // 使用するターゲット
        useTarget = (_renderTarget == SelectTarget_ViewFrameBuffer) ? &_renderBuffer : &refModel.GetRenderBuffer();

        if (!useTarget->IsValid())
        {// 描画ターゲット内部未作成の場合はここで作成
            int width = LAppDelegate::GetInstance()->GetWindowWidth();
            int height = LAppDelegate::GetInstance()->GetWindowHeight();

            // モデル描画キャンバス
            useTarget->CreateOffscreenSurface(static_cast<csmUint32>(width), static_cast<csmUint32>(height));
        }

        // レンダリング開始
        useTarget->BeginDraw();
        useTarget->Clear(_clearColor[0], _clearColor[1], _clearColor[2], _clearColor[3]); // 背景クリアカラー
    }
}

void LAppView::PostModelDraw(LAppModel &refModel)
{
    // 別のレンダリングターゲットへ向けて描画する場合の使用するフレームバッファ
    Csm::Rendering::CubismOffscreenSurface_OpenGLES2* useTarget = NULL;

    if (_renderTarget != SelectTarget_None)
    {// 別のレンダリングターゲットへ向けて描画する場合

        // 使用するターゲット
        useTarget = (_renderTarget == SelectTarget_ViewFrameBuffer) ? &_renderBuffer : &refModel.GetRenderBuffer();

        // レンダリング終了
        useTarget->EndDraw();

        // LAppViewの持つフレームバッファを使うなら、スプライトへの描画はここ
        if (_renderTarget == SelectTarget_ViewFrameBuffer && _renderSprite)
        {
            const GLfloat uvVertex[] =
            {
                1.0f, 1.0f,
                0.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,
            };

            _renderSprite->SetColor(1.0f, 1.0f, 1.0f, GetSpriteAlpha(0));
            _renderSprite->RenderImmidiate(useTarget->GetColorBuffer(), uvVertex);
        }
    }
}

void LAppView::SwitchRenderingTarget(SelectTarget targetType)
{
    _renderTarget = targetType;
}

void LAppView::SetRenderTargetClearColor(float r, float g, float b)
{
    _clearColor[0] = r;
    _clearColor[1] = g;
    _clearColor[2] = b;
}

float LAppView::GetSpriteAlpha(int assign) const
{
    // assignの数値に応じて適当に決定
    float alpha = 0.25f + static_cast<float>(assign) * 0.5f; // サンプルとしてαに適当な差をつける
    if (alpha > 1.0f)
    {
        alpha = 1.0f;
    }
    if (alpha < 0.1f)
    {
        alpha = 0.1f;
    }

    return alpha;
}
