/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

#pragma once

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <jni.h>
#include "LAppAllocator.hpp"
#include <string>

class LAppView;
class LAppTextureManager;

class LAppModelParameters
{
public:
    // モデル
    float modelScale = 1.0f;               ///< モデル表示倍率
    float modelTranslateX = 0.0f;          ///< モデル表示位置X
    float modelTranslateY = 0.0f;          ///< モデル表示位置Y
    bool breathEnabled = true;            ///< 呼吸アニメを切り替える
    // 顔
    float faceX = 0.0f;                    ///< X軸の顔の向きの値
    float faceY = 0.0f;                    ///< Y軸の顔の向きの値
    float faceZ = 0.0f;                    ///< Z軸の顔の向きの値
    // 目
    bool autoBlinkEyesEnabled = false;     ///< 自動まばたきを切り替える
    bool manualEyeBallEnabled = false;
    float eyeLOpen = 1.0f;
    float eyeROpen = 1.0f;
    float eyeBallX = 0.0f;                  ///< 目玉表示位置X
    float eyeBallY = 0.0f;                  ///< 目玉表示位置Y
    // 口
    float mouthForm = 1.0f;
    float mouthOpenY = 0.0f;
    // アニメーション
    bool changeMotion = false;              ///< モーション切り替えフラグ
    bool stopMotion = false;                ///< モーション再生停止
    bool resetMotion = false;               ///< モーション再生回復
    std::string nextMotionGroup = "";       ///< 次のモーショングループ
    Csm::csmInt32 nextMotionIndex = 0;      ///< 次のモーションインデックス

    bool changeExpression = false;
    bool stopExpression = false;
    std::string nextExpressionName = "";
};

/**
* @brief   アプリケーションクラス。
*   Cubism SDK の管理を行う。
*/
class LAppDelegate
{
public:
    /**
    * @brief   クラスのインスタンス（シングルトン）を返す。<br>
    *           インスタンスが生成されていない場合は内部でインスタンを生成する。
    *
    * @return  クラスのインスタンス
    */
    static LAppDelegate* GetInstance();

    /**
    * @brief   クラスのインスタンス（シングルトン）を解放する。
    *
    */
    static void ReleaseInstance();

    /**
    * @brief JavaのActivityのOnStart()のコールバック関数。
    */
    void OnStart();

    /**
    * @brief JavaのActivityのOnPause()のコールバック関数。
    */
    void OnPause();

    /**
    * @brief JavaのActivityのOnStop()のコールバック関数。
    */
    void OnStop();

    /**
    * @brief JavaのActivityのOnDestroy()のコールバック関数。
    */
    void OnDestroy();

    /**
    * @brief   JavaのGLSurfaceviewのOnSurfaceCreate()のコールバック関数。
    */
    void OnSurfaceCreate();

    /**
     * @brief JavaのGLSurfaceviewのOnSurfaceChanged()のコールバック関数。
     * @param width
     * @param height
     */
    void OnSurfaceChanged(float width, float height);

    /**
    * @brief   実行処理。
    */
    void Run();

    void ModelChangeToName(const char* modelName);

    void ModelChangeTo(const char* modelPath, const char* modelJsonFileName);

    void ApplyMotion(const char* motionGroup, signed int index);

    void StopMotion();

    void ResetMotion();

    void ApplyExpression(const char* expressionName);

    void StopExpression();

    void ModelResize(float scale);

    void ModelTranslateX(float x);

    void ModelTranslateY(float y);

    void ModelTranslateFace(float x, float y, float z);

    void ModelBreath(bool enabled);

    void ModelAutoBlinkEyes(bool enabled);

    void ModelEyeLOpen(float value);

    void ModelEyeROpen(float value);

    void ModelMouthForm(float value);

    void ModelMouthOpenY(float value);

    void ModelEyeBallX(float value);

    void ModelEyeBallY(float value);

    void ModelManualEyeBall(float value);

    /**
    * @brief シェーダーを登録する。
    */
    GLuint CreateShader();

    /**
    * @brief テクスチャマネージャーの取得
    */
    LAppTextureManager* GetTextureManager() { return _textureManager; }

    /**
    * @brief ウインドウ幅の設定
    */
    int GetWindowWidth() { return _width; }

    /**
    * @brief ウインドウ高さの取得
    */
    int GetWindowHeight() { return _height; }

    /**
    * @brief   アプリケーションを非アクティブにする。
    */
    void DeActivateApp() { _isActive = false; }

    /**
    * @brief   View情報を取得する。
    */
    LAppView* GetView() { return _view; }

private:
    /**
    * @brief   コンストラクタ
    */
    LAppDelegate();

    /**
    * @brief   デストラクタ
    */
    ~LAppDelegate();

    /**
    * @brief   Cubism SDK の初期化
    */
    void InitializeCubism();

    LAppAllocator _cubismAllocator;              ///< Cubism SDK Allocator
    Csm::CubismFramework::Option _cubismOption;  ///< Cubism SDK Option
    LAppTextureManager* _textureManager;         ///< テクスチャマネージャー
    LAppView* _view;                             ///< View情報
    int _width;                                  ///< Windowの幅
    int _height;                                 ///< windowの高さ
    int _SceneIndex;                             ///< モデルシーンインデックス
    bool _captured;                              ///< クリックしているか
    bool _isActive;                              ///< アプリがアクティブ状態なのか
    float _mouseY;                               ///< マウスY座標
    float _mouseX;                               ///< マウスX座標
};
