/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

#pragma once

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <Math/CubismMatrix44.hpp>
#include <Math/CubismViewMatrix.hpp>
#include "CubismFramework.hpp"
#include "LAppDelegate.hpp"
#include <Rendering/OpenGL/CubismOffscreenSurface_OpenGLES2.hpp>
#include <string>

class TouchManager;
class LAppSprite;
class LAppModel;

/**
* @brief 描画クラス
*/
class LAppView
{
public:

    /**
     * @brief LAppModelのレンダリング先
     */
    enum SelectTarget
    {
        SelectTarget_None,                ///< デフォルトのフレームバッファにレンダリング
        SelectTarget_ModelFrameBuffer,    ///< LAppModelが各自持つフレームバッファにレンダリング
        SelectTarget_ViewFrameBuffer,     ///< LAppViewの持つフレームバッファにレンダリング
    };

    /**
    * @brief コンストラクタ
    */
    LAppView();

    /**
    * @brief デストラクタ
    */
    ~LAppView();

    /**
    * @brief 初期化する。
    */
    void Initialize();

    void Resize(float scale);

    void TranslateX(float x);

    void TranslateY(float y);

    void TranslateFace(float x, float y, float z);

    void Breath(bool enabled);

    void AutoBlinkEyes(bool enabled);

    void ModelEyeLOpen(float value);

    void ModelEyeROpen(float value);

    void ModelMouthForm(float value);

    void ModelMouthOpenY(float value);

    void ModelEyeBallX(float value);

    void ModelEyeBallY(float value);

    void ModelManualEyeBall(float value);

    /**
    * @brief 描画する。
    */
    void Render();

    void ChangeModelTo(std::string modelPath, std::string modelJsonFileName);

    void ApplyMotion(std::string motionGroup, Csm::csmInt32 index);

    void StopMotion();

    void ResetMotion();

    void ApplyExpression(const char* expressionName);

    void StopExpression();

    /**
    * @brief シェーダーの初期化を行う。
    */
    void InitializeShader();

    /**
    * @brief X座標をView座標に変換する。
    *
    * @param[in]       deviceX            デバイスX座標
    */
    float TransformViewX(float deviceX) const;

    /**
    * @brief Y座標をView座標に変換する。
    *
    * @param[in]       deviceY            デバイスY座標
    */
    float TransformViewY(float deviceY) const;

    /**
    * @brief X座標をScreen座標に変換する。
    *
    * @param[in]       deviceX            デバイスX座標
    */
    float TransformScreenX(float deviceX) const;

    /**
    * @brief Y座標をScreen座標に変換する。
    *
    * @param[in]       deviceY            デバイスY座標
    */
    float TransformScreenY(float deviceY) const;

    /**
     * @brief   モデル1体を描画する直前にコールされる
     */
    void PreModelDraw(LAppModel &refModel);

    /**
     * @brief   モデル1体を描画した直後にコールされる
     */
    void PostModelDraw(LAppModel &refModel);

    /**
     * @brief   別レンダリングターゲットにモデルを描画するサンプルで
     *           描画時のαを決定する
     */
    float GetSpriteAlpha(int assign) const;

    /**
     * @brief レンダリング先を切り替える
     */
    void SwitchRenderingTarget(SelectTarget targetType);

    /**
     * @brief レンダリング先をデフォルト以外に切り替えた際の背景クリア色設定
     * @param[in]   r   赤(0.0~1.0)
     * @param[in]   g   緑(0.0~1.0)
     * @param[in]   b   青(0.0~1.0)
     */
    void SetRenderTargetClearColor(float r, float g, float b);

private:
    Csm::CubismMatrix44* _deviceToScreen;    ///< デバイスからスクリーンへの行列
    Csm::CubismViewMatrix* _viewMatrix;      ///< viewMatrix
    GLuint _programId;                       ///< シェーダID
    bool _changeModel;                       ///< モデル切り替えフラグ
    std::string _nextModelPath;              ///< 次のモデルパス
    std::string _nextModelJsonFileName;      ///< 次のモデルJsonファイル名前
    LAppModelParameters _modelParameters;    ///< モデルのパラメータ

    // レンダリング先を別ターゲットにする方式の場合に使用
    LAppSprite* _renderSprite;                                      ///< モードによっては_renderBufferのテクスチャを描画
    Csm::Rendering::CubismOffscreenSurface_OpenGLES2 _renderBuffer;   ///< モードによってはCubismモデル結果をこっちにレンダリング
    SelectTarget _renderTarget;     ///< レンダリング先の選択肢
    float _clearColor[4];           ///< レンダリングターゲットのクリアカラー
};
