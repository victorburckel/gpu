#include "NBodyCPU.h"
#include "ParticleCpu.h"
#include "Resources.h"

#include <cinder/app/AppNative.h>
#include <cinder/gl/gl.h>
#include <cinder/params/Params.h>
#include <cinder/gl/Vbo.h>
#include <cinder/gl/GlslProg.h>
#include <cinder/Camera.h>

#include <random>

using namespace ci;
using namespace ci::app;
using namespace std;
using namespace concurrency::graphics;

namespace {
	void addDynamicCustomAttribute( gl::VboMesh::Layout& layout, gl::VboMesh::Layout::CustomAttr type )
	{
		layout.mCustomDynamic.push_back( std::make_pair( type, 0u ) );
	}

	class VBOMeshExt;

	typedef std::shared_ptr< VBOMeshExt > VBOMeshExtRef;

	class VBOMeshExt : public gl::VboMesh
	{
	public:
		VBOMeshExt( size_t numVertices, size_t numIndices, Layout layout, GLenum primitiveType ) : VboMesh( numVertices, numIndices, layout, primitiveType )
		{}

		void bufferDynamicCustomAttributeData( size_t customAttribute, const std::vector< Vec3f >& v )
		{
			if( mObj->mLayout.mCustomDynamic[ customAttribute ].first != gl::VboMesh::Layout::CUSTOM_ATTR_FLOAT3 )
				throw;

			getDynamicVbo().bufferSubData( mObj->mLayout.mCustomDynamic[ customAttribute ].second, sizeof( Vec3f )* v.size(), &v[ 0 ] );
		}

		static VBOMeshExtRef create( size_t numVertices, size_t numIndices, Layout layout, GLenum primitiveType )
		{
			return std::make_shared< VBOMeshExt >( numVertices, numIndices, layout, primitiveType );
		}
	};
}

class NBodyCPUApp : public AppNative
{
public:
	NBodyCPUApp();

	virtual void setup() override;
	virtual void prepareSettings( Settings* ) override;
	virtual void update() override;
	virtual void draw() override;

private:
	params::InterfaceGl _params;
	float _fps = 0.0f;
	double _lastElapsed;

	// Particles
	int _numParticles = 1000;
	static const int _minParticles = 100;
	static const int _maxParticles = ( 15 * 1024 );
	static const int _stepParticles = 100;

	__declspec( align( SSE_ALIGNMENTBOUNDARY ) ) std::vector< ParticleCpu > _particlesOld;
	__declspec( align( SSE_ALIGNMENTBOUNDARY ) ) std::vector< ParticleCpu > _particlesNew;

	// Number of particles added for each slider tick
	static const int _particleNumStepSize = 256;
	// Separation between the two clusters
	const float _spread = 400.0f;

	// Computation type CPU Single Core, CPU Multi Core
	int _parameterComputeType = 0;
	ComputeType _cumputeType = ComputeType::CpuSingle;
	std::unique_ptr< INBodyCpu > _nBody;

	// Rendering
	CameraPersp _camera;

	std::vector< Vec3f > _positions;
	VBOMeshExtRef _mesh;
	gl::GlslProgRef _shader;

	void loadParticles();
};

NBodyCPUApp::NBodyCPUApp()
	: _particlesOld( _maxParticles )
	, _particlesNew( _maxParticles )
	, _positions( _maxParticles )
{}

namespace {
	template< typename T >
	inline float_3 polarToCartesian( T r, T theta, T phi )
	{
		return float_3( r * sin( theta ) * cos( phi ), r * sin( theta ) * sin( phi ), r * cos( theta ) );
	}

	void loadClusterParticles( ParticleCpu* const pParticles, float_3 center, float_3 velocity, float spread, int numParticles )
	{
		std::random_device rd;
		std::default_random_engine engine( rd() );
		std::uniform_real_distribution< float > randRadius( 0.0f, spread );
		std::uniform_real_distribution< float > randTheta( -1.0f, 1.0f );
		std::uniform_real_distribution< float > randPhi( 0.0f, 2.0f * static_cast<float>( std::_Pi ) );

		std::for_each( pParticles, pParticles + numParticles, [ =, &engine, &randRadius, &randTheta, &randPhi ]( ParticleCpu& p ) {
			float_3 delta = polarToCartesian( randRadius( engine ), acos( randTheta( engine ) ), randPhi( engine ) );
			p.pos = center + delta;
			p.vel = velocity;
			p.acc = 0.0f;
		} );
	}

	std::unique_ptr< INBodyCpu > getNBody( ComputeType cumputeType )
	{
		switch( cumputeType )
		{
		case ComputeType::CpuSingle:
			return std::make_unique< NBodySimpleSingleCore >();
		case ComputeType::CpuMulti:
			return std::make_unique< NBodySimpleMultiCore >();
		default:
			throw std::runtime_error( "Bad computeType" );
		}
	}
}

void NBodyCPUApp::loadParticles()
{
	const float centerSpread = _spread * 0.50f;
	for( size_t i = 0; i < _maxParticles; i += _particleNumStepSize )
	{
		loadClusterParticles( &_particlesOld[ i ], float_3( centerSpread, 0.0f, 0.0f ), float_3( 0, 0, -20 ), _spread, _particleNumStepSize / 2 );
		loadClusterParticles( &_particlesOld[ i + _particleNumStepSize / 2 ], float_3( -centerSpread, 0.0f, 0.0f ), float_3( 0, 0, 20 ), _spread, ( _particleNumStepSize + 1 ) / 2 );
	}
}

void NBodyCPUApp::setup()
{
	// NBody setup
	_nBody = getNBody( _cumputeType );

	// Particles setup
	loadParticles();

	// Parameters setup
	_params = { "Parameters", Vec2i( 225, 200 ) };
	_params.addParam( "Particles number", &_numParticles )
		.min( static_cast<float>( _minParticles ) )
		.max( static_cast<float>( _maxParticles ) )
		.step( static_cast<float>( _stepParticles ) );
	_params.addButton( "Reset particles", [ this ](){ loadParticles(); } );
	_params.addParam( "Type", { "CPU Single Core", "CPU Multi Core" }, &_parameterComputeType );
	_params.addParam( "FPS", &_fps, true );

	// VBO creation
	try
	{
		gl::VboMesh::Layout layout;

		// Positions
		addDynamicCustomAttribute( layout, gl::VboMesh::Layout::CUSTOM_ATTR_FLOAT3 );

		_mesh = VBOMeshExt::create( _maxParticles, 0u, layout, GL_POINTS );
		_mesh->setCustomDynamicLocation( 0u, 0 );
	}
	catch( ... )
	{
		app::console() << "VBO creation error" << std::endl;
	}

	// Shader compilation
	try
	{
		_shader = gl::GlslProg::create( loadResource( VERTEX_SHADER ), loadResource( FRAGMENT_SHADER ) );
	}
	catch( const gl::GlslProgCompileExc &e )
	{
		app::console() << "Shader compile error: " << e.what() << std::endl;
	}
	catch( ... )
	{
		app::console() << "Shader compile error" << std::endl;
	}

	// Camera setup
	_camera.lookAt( { -_spread * 2, _spread * 4, -_spread * 3 }, { 0.0f, 0.0f, 0.0f } );
	_camera.setPerspective( 45, getWindowAspectRatio(), 10.0f, 500000.0f );

	// Uniforms
	_shader->bind();
	_shader->uniform( "color", Vec3f( 1.0f, 0.0f, 0.0f ) );
	_shader->uniform( "matrix", _camera.getProjectionMatrix() * _camera.getModelViewMatrix() );
	_shader->unbind();

	_lastElapsed = getElapsedSeconds();
}

void NBodyCPUApp::prepareSettings( Settings* settings )
{
	settings->setFrameRate( 60.0f );
}

void NBodyCPUApp::update()
{
	const auto elapsed = getElapsedSeconds();
	if( static_cast<ComputeType>( _parameterComputeType ) != _cumputeType )
	{
		_cumputeType = static_cast<ComputeType>( _parameterComputeType );
		_nBody = getNBody( _cumputeType );
	}

	_nBody->integrate( &_particlesOld[ 0 ], &_particlesNew[ 0 ], _numParticles, static_cast<float>( elapsed - _lastElapsed ) );
	_particlesNew.swap( _particlesOld );
	_fps = getAverageFps();
	_lastElapsed = elapsed;
}

void NBodyCPUApp::draw()
{
	// clear out the window with black
	gl::clear( Color( 0, 0, 0 ) );
	// draw particles
	std::transform( _particlesOld.cbegin(), _particlesOld.cbegin() + _maxParticles, _positions.begin(), []( const ParticleCpu& p ){  return Vec3f( p.pos.x, p.pos.y, p.pos.z ); } );
	_mesh->bufferDynamicCustomAttributeData( 0u, _positions );

	_shader->bind();
	gl::drawArrays( _mesh, 0, _numParticles );
	_shader->unbind();

	// draw parameters
	_params.draw();
}

CINDER_APP_NATIVE( NBodyCPUApp, RendererGl )
