/**
 * Copyright(c) Live2D Inc. All rights reserved.
 *
 * Use of this source code is governed by the Live2D Open Software license
 * that can be found at https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html.
 */

#include "LAppDelegate.hpp"
#include <iostream>
#include <GLES2/gl2.h>
#include "LAppView.hpp"
#include "LAppPal.hpp"
#include "LAppDefine.hpp"
#include "LAppLive2DManager.hpp"
#include "LAppTextureManager.hpp"
#include "JniBridgeC.hpp"

using namespace Csm;
using namespace std;
using namespace LAppDefine;

namespace {
    LAppDelegate* s_instance = NULL;
}

LAppDelegate* LAppDelegate::GetInstance()
{
    if (s_instance == NULL)
    {
        s_instance = new LAppDelegate();
    }

    return s_instance;
}

void LAppDelegate::ReleaseInstance()
{
    if (s_instance != NULL)
    {
        delete s_instance;
    }

    s_instance = NULL;
}


void LAppDelegate::OnStart()
{
    _textureManager = new LAppTextureManager();
    _view = new LAppView();
    LAppPal::UpdateTime();
}

void LAppDelegate::OnPause()
{
    _SceneIndex = LAppLive2DManager::GetInstance()->GetSceneIndex();
}

void LAppDelegate::OnStop()
{
    // リソースを解放
    LAppLive2DManager::ReleaseInstance();

    if (_view)
    {
        delete _view;
        _view = NULL;
    }
    if (_textureManager)
    {
        delete _textureManager;
        _textureManager = NULL;
    }

    CubismFramework::Dispose();
}

void LAppDelegate::OnDestroy()
{
    ReleaseInstance();
}

void LAppDelegate::Run()
{
    // 時間更新
    LAppPal::UpdateTime();

    // 画面の初期化
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearDepthf(1.0f);

    //描画更新
    if (_view != NULL)
    {
        _view->Render();
    }

    if(_isActive == false)
    {
        JniBridgeC::MoveTaskToBack();
    }
}

void LAppDelegate::OnSurfaceCreate()
{
    //テクスチャサンプリング設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    //透過設定
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //Initialize cubism
    CubismFramework::Initialize();

    _view->InitializeShader();
}

void LAppDelegate::OnSurfaceChanged(float width, float height)
{
    glViewport(0, 0, width, height);
    _width = width;
    _height = height;

    //AppViewの初期化
    _view->Initialize();

    //load model
//    if (LAppLive2DManager::GetInstance()->GetSceneIndex() != _SceneIndex)
//    {
//        LAppLive2DManager::GetInstance()->ChangeScene(_SceneIndex);
//    }

    _isActive = true;
}

LAppDelegate::LAppDelegate():
    _cubismOption(),
    _captured(false),
    _SceneIndex(0),
    _mouseX(0.0f),
    _mouseY(0.0f),
    _isActive(true),
    _textureManager(NULL),
    _view(NULL)
{
    // Setup Cubism
    _cubismOption.LogFunction = LAppPal::PrintMessage;
    _cubismOption.LoggingLevel = LAppDefine::CubismLoggingLevel;
    CubismFramework::CleanUp();
    CubismFramework::StartUp(&_cubismAllocator, &_cubismOption);
}

LAppDelegate::~LAppDelegate()
{
}

void LAppDelegate::ModelChangeToName(const char* modelName)
{
    // model3.jsonのパスを決定する.
    // ディレクトリ名とmodel3.jsonの名前を一致させておくこと.
    std::string modelNameStr(modelName);
    std::string modelPath = modelNameStr + "/";
    std::string modelJsonName = modelNameStr;
    modelJsonName += ".model3.json";
    LAppDelegate::ModelChangeTo(modelPath.c_str(), modelJsonName.c_str());
}

void LAppDelegate::ModelChangeTo(const char* modelPath, const char* modelJsonFileName)
{
    if (_view != NULL)
    {
        _view->ChangeModelTo(modelPath, modelJsonFileName);
    }
    // LAppLive2DManager::GetInstance()->ChangeSceneTo(modelname);
    // LAppLive2DManager::GetInstance()->ChangeScene(2);
}

void LAppDelegate::ApplyMotion(const char* motionGroup, signed int index)
{
    if (_view != NULL)
    {
        _view->ApplyMotion(motionGroup, index);
    }
}

void LAppDelegate::StopMotion()
{
    if (_view != NULL)
    {
        _view->StopMotion();
    }
}

void LAppDelegate::ResetMotion()
{
    if (_view != NULL)
    {
        _view->ResetMotion();
    }
}

void LAppDelegate::ApplyExpression(const char *expressionName)
{
    if (_view != NULL)
    {
        _view->ApplyExpression(expressionName);
    }
}

void LAppDelegate::StopExpression()
{
    if (_view != NULL)
    {
        _view->StopExpression();
    }
}

void LAppDelegate::ModelResize(float scale)
{
    if (_view != NULL)
    {
        _view->Resize(scale);
    }
}

void LAppDelegate::ModelTranslateX(float x)
{
    if (_view != NULL)
    {
        _view->TranslateX(x);
    }
}

void LAppDelegate::ModelTranslateY(float y)
{
    if (_view != NULL)
    {
        _view->TranslateY(y);
    }
}

void LAppDelegate::ModelTranslateFace(float x, float y, float z)
{
    if (_view != NULL)
    {
        _view->TranslateFace(x, y, z);
    }
}

void LAppDelegate::ModelBreath(bool enabled)
{
    if (_view != NULL)
    {
        _view->Breath(enabled);
    }
}

void LAppDelegate::ModelAutoBlinkEyes(bool enabled)
{
    if (_view != NULL)
    {
        _view->AutoBlinkEyes(enabled);
    }
}

void LAppDelegate::ModelEyeLOpen(float value)
{
    if (_view != NULL)
    {
        _view->ModelEyeLOpen(value);
    }
}

void LAppDelegate::ModelEyeROpen(float value)
{
    if (_view != NULL)
    {
        _view->ModelEyeROpen(value);
    }
}

void LAppDelegate::ModelMouthForm(float value)
{
    if (_view != NULL)
    {
        _view->ModelMouthForm(value);
    }
}

void LAppDelegate::ModelMouthOpenY(float value)
{
    if (_view != NULL)
    {
        _view->ModelMouthOpenY(value);
    }
}

void LAppDelegate::ModelEyeBallX(float value)
{
    if (_view != NULL)
    {
        _view->ModelEyeBallX(value);
    }
}

void LAppDelegate::ModelEyeBallY(float value)
{
    if (_view != NULL)
    {
        _view->ModelEyeBallY(value);
    }
}

void LAppDelegate::ModelManualEyeBall(float value)
{
    if (_view != NULL)
    {
        _view->ModelManualEyeBall(value);
    }
}

GLuint LAppDelegate::CreateShader()
{
    //バーテックスシェーダのコンパイル
    GLuint vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexShader =
        "#version 100\n"
        "attribute vec3 position;"
        "attribute vec2 uv;"
        "varying vec2 vuv;"
        "void main(void){"
        "    gl_Position = vec4(position, 1.0);"
        "    vuv = uv;"
        "}";
    glShaderSource(vertexShaderId, 1, &vertexShader, NULL);
    glCompileShader(vertexShaderId);

    //フラグメントシェーダのコンパイル
    GLuint fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fragmentShader =
        "#version 100\n"
        "precision mediump float;"
        "varying vec2 vuv;"
        "uniform sampler2D texture;"
        "uniform vec4 baseColor;"
        "void main(void){"
        "    gl_FragColor = texture2D(texture, vuv) * baseColor;"
        "}";
    glShaderSource(fragmentShaderId, 1, &fragmentShader, NULL);
    glCompileShader(fragmentShaderId);

    //プログラムオブジェクトの作成
    GLuint programId = glCreateProgram();
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragmentShaderId);

    // リンク
    glLinkProgram(programId);

    glUseProgram(programId);

    return programId;
}
