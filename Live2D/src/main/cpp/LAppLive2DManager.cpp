/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

#include "LAppLive2DManager.hpp"
#include <string>
#include <GLES2/gl2.h>
#include <Rendering/CubismRenderer.hpp>
#include "LAppPal.hpp"
#include "LAppDefine.hpp"
#include "LAppDelegate.hpp"
#include "LAppModel.hpp"
#include "LAppView.hpp"
#include "JniBridgeC.hpp"

using namespace Csm;
using namespace LAppDefine;
using namespace std;

namespace {
    LAppLive2DManager* s_instance = NULL;

    void FinishedMotion(ACubismMotion* self)
    {
        LAppPal::PrintLog("Motion Finished: %x", self);
    }
}

LAppLive2DManager* LAppLive2DManager::GetInstance()
{
    if (s_instance == NULL)
    {
        s_instance = new LAppLive2DManager();
    }

    return s_instance;
}

void LAppLive2DManager::ReleaseInstance()
{
    if (s_instance != NULL)
    {
        delete s_instance;
    }

    s_instance = NULL;
}

LAppLive2DManager::LAppLive2DManager()
    : _viewMatrix(NULL)
    , _sceneIndex(0)
{
//    ChangeScene(_sceneIndex);
}

LAppLive2DManager::~LAppLive2DManager()
{
    ReleaseAllModel();
}

void LAppLive2DManager::ReleaseAllModel()
{
    for (csmUint32 i = 0; i < _models.GetSize(); i++)
    {
        delete _models[i];
    }

    _models.Clear();
}

LAppModel* LAppLive2DManager::GetModel(csmUint32 no) const
{
    if (no < _models.GetSize())
    {
        return _models[no];
    }

    return NULL;
}

void LAppLive2DManager::OnUpdate(LAppModelParameters modelParameters) const
{
    CubismMatrix44 projection;
    int width = LAppDelegate::GetInstance()->GetWindowWidth();
    int height = LAppDelegate::GetInstance()->GetWindowHeight();
    projection.Scale(1.0f * modelParameters.modelScale, static_cast<float>(width) / static_cast<float>(height) * modelParameters.modelScale);
    projection.Translate(modelParameters.modelTranslateX, modelParameters.modelTranslateY);

    if (_viewMatrix != NULL)
    {
        projection.MultiplyByMatrix(_viewMatrix);
    }

    const CubismMatrix44    saveProjection = projection;
    csmUint32 modelCount = _models.GetSize();
    for (csmUint32 i = 0; i < modelCount; ++i)
    {
        LAppModel* model = GetModel(i);
        projection = saveProjection;

        // モデル1体描画前コール
        LAppDelegate::GetInstance()->GetView()->PreModelDraw(*model);

        model->Update(modelParameters);
        model->Draw(projection);///< 参照渡しなのでprojectionは変質する

        // モデル1体描画後コール
        LAppDelegate::GetInstance()->GetView()->PostModelDraw(*model);
    }
}

void LAppLive2DManager::ChangeModelTo(std::string modelPath, std::string modelJsonFileName)
{
    // モデルパスの「/」は必要な、モデルJsonファイル名前は「xxx.model3.json」
    if (DebugLogEnable)
    {
        LAppPal::PrintLog("[APP]model load: %s (Dir: %s)", modelJsonFileName.c_str(), modelPath.c_str());
    }

    LAppModel* newModel = new LAppModel();

    ReleaseAllModel();
    if (newModel->LoadAssets(modelPath.c_str(), modelJsonFileName.c_str())) {
        // ReleaseAllModel();
        _models.PushBack(newModel);

        /*
         * モデル半透明表示を行うサンプルを提示する。
         * ここでUSE_RENDER_TARGET、USE_MODEL_RENDER_TARGETが定義されている場合
         * 別のレンダリングターゲットにモデルを描画し、描画結果をテクスチャとして別のスプライトに張り付ける。
         */
        {
#if defined(USE_RENDER_TARGET)
            // LAppViewの持つターゲットに描画を行う場合、こちらを選択
        LAppView::SelectTarget useRenderTarget = LAppView::SelectTarget_ViewFrameBuffer;
#elif defined(USE_MODEL_RENDER_TARGET)
            // 各LAppModelの持つターゲットに描画を行う場合、こちらを選択
        LAppView::SelectTarget useRenderTarget = LAppView::SelectTarget_ModelFrameBuffer;
#else
            // デフォルトのメインフレームバッファへレンダリングする(通常)
            LAppView::SelectTarget useRenderTarget = LAppView::SelectTarget_None;
#endif

#if defined(USE_RENDER_TARGET) || defined(USE_MODEL_RENDER_TARGET)
            // モデル個別にαを付けるサンプルとして、もう1体モデルを作成し、少し位置をずらす
        _models.PushBack(new LAppModel());
        _models[1]->LoadAssets(modelPath.c_str(), modelJsonName.c_str());
        _models[1]->GetModelMatrix()->TranslateX(0.2f);
#endif

            LAppDelegate::GetInstance()->GetView()->SwitchRenderingTarget(useRenderTarget);

            // 別レンダリング先を選択した際の背景クリア色
            float clearColor[3] = { 1.0f, 1.0f, 1.0f };
            LAppDelegate::GetInstance()->GetView()->SetRenderTargetClearColor(clearColor[0], clearColor[1], clearColor[2]);
        }
        LAppPal::PrintLog("[APP]bili L2D glGetError (INIT): %d", glGetError());
        JniBridgeC::OnLoadDone();
    } else {
        delete newModel;
        if (DebugLogEnable)
        {
            LAppPal::PrintLog("[APP]model load: %s (Dir: %s), failed.", modelJsonFileName.c_str(), modelPath.c_str());
        }
        JniBridgeC::OnLoadError();
    }
}

csmUint32 LAppLive2DManager::GetModelNum() const
{
    return _models.GetSize();
}
