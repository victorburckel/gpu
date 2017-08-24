#include "ParticlesAMP.h"
#include "Resources.h"
#include "NBodyAMP.h"

#include <cinder/app/AppNative.h>
#include <cinder/gl/gl.h>
#include <cinder/params/Params.h>
#include <cinder/gl/Vbo.h>
#include <cinder/gl/GlslProg.h>
#include <cinder/Camera.h>

#include <random>
#include <codecvt>

using namespace ci;
using namespace ci::app;
using namespace std;

using namespace concurrency;
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

		void bufferDynamicCustomAttributeData( size_t customAttribute, const std::vector< float_3 >& v )
		{
			if( mObj->mLayout.mCustomDynamic[ customAttribute ].first != gl::VboMesh::Layout::CUSTOM_ATTR_FLOAT3 )
				throw;

			getDynamicVbo().bufferSubData( mObj->mLayout.mCustomDynamic[ customAttribute ].second, sizeof( float_3 )* v.size(), &v[ 0 ] );
		}

		static VBOMeshExtRef create( size_t numVertices, size_t numIndices, Layout layout, GLenum primitiveType )
		{
			return std::make_shared< VBOMeshExt >( numVertices, numIndices, layout, primitiveType );
		}
	};
}

class NBodyAMPApp : public AppNative
{
public:
	NBodyAMPApp();

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
	static const int _maxParticles = ( 20 * 1024 );
	static const int _stepParticles = 100;

	std::vector< std::unique_ptr< TaskData > > _tasks;
	ParticlesCpu _particlesCpu;

	// Number of particles added for each slider tick
	static const int _particleNumStepSize = 256;
	// Separation between the two clusters
	const float _spread = 400.0f;

	// Computation type CPU Single Core, CPU Multi Core
	int _parameterTask = 0;
	size_t _task = 0;
	std::unique_ptr< INBodyAMP > _nBody;

	// Rendering
	CameraPersp _camera;

	std::vector< Vec3f > _positions;
	VBOMeshExtRef _mesh;
	gl::GlslProgRef _shader;

	void loadParticles();
	void uploadData();
};

NBodyAMPApp::NBodyAMPApp()
	: _particlesCpu( _maxParticles )
{}

namespace {
	class IsAmpAccelerator
	{
	public:
		IsAmpAccelerator( bool includeWarp ) : _includeWarp( includeWarp )
		{}

		bool operator() ( const concurrency::accelerator& a ) const
		{
			return ( a.is_emulated || ( ( a.device_path.compare( concurrency::accelerator::direct3d_warp ) == 0 ) && !_includeWarp ) );
		}

	private:
		const bool _includeWarp;
	};

	template<typename Func>
	std::vector< accelerator > getAccelerators( Func filter )
	{
		auto accls = accelerator::get_all();
		accls.erase( std::remove_if( accls.begin(), accls.end(), filter ), accls.end() );
		return accls;
	}

	std::vector< accelerator > getGpuAccelerators()
	{
		return getAccelerators( IsAmpAccelerator( false ) );
	}

	std::vector< std::unique_ptr< TaskData > > createTasks( int numParticles )
	{
		const auto gpuAccelerators = getGpuAccelerators();
		std::vector< std::unique_ptr< TaskData > > tasks;
		tasks.reserve( gpuAccelerators.size() );

		if( !gpuAccelerators.empty() )
		{
			//  All other GPUs are associated with their default view.
			std::for_each( gpuAccelerators.cbegin(), gpuAccelerators.cend(), [ =, &tasks ]( const accelerator& d )
			{
				tasks.push_back( std::make_unique< TaskData >( numParticles, d.default_view, d ) );
			} );
		}

		if( tasks.empty() )
		{
			accelerator a( accelerator::default_accelerator );
			tasks.push_back( std::make_unique< TaskData >( numParticles, a.default_view, a ) );
		}

		return tasks;
	}

	template<typename T>
	inline float_3 polarToCartesian( T r, T theta, T phi )
	{
		return float_3( r * sin( theta ) * cos( phi ), r * sin( theta ) * sin( phi ), r * cos( theta ) );
	}

	void loadClusterParticles( ParticlesCpu& particles, int offset, int size, float_3 center, float_3 velocity, float spread )
	{
		std::random_device rd;
		std::default_random_engine engine( rd() );
		std::uniform_real_distribution< float > randRadius( 0.0f, spread );
		std::uniform_real_distribution< float > randTheta( -1.0f, 1.0f );
		std::uniform_real_distribution< float > randPhi( 0.0f, 2.0f * static_cast<float>( std::_Pi ) );

		for( int i = offset; i < ( offset + size ); ++i )
		{
			float_3 delta = polarToCartesian( randRadius( engine ), acos( randTheta( engine ) ), randPhi( engine ) );
			particles.pos[ i ] = center + delta;
			particles.vel[ i ] = velocity;
		};
	}
}

void NBodyAMPApp::loadParticles()
{
	const float centerSpread = _spread * 0.50f;

	for( int i = 0; i < _maxParticles; i += _particleNumStepSize )
	{
		loadClusterParticles( _particlesCpu, i, ( _particleNumStepSize / 2 ), float_3( centerSpread, 0.0f, 0.0f ), float_3( 0, 0, -20 ), _spread );
		loadClusterParticles( _particlesCpu, ( i + _particleNumStepSize / 2 ), ( ( _particleNumStepSize + 1 ) / 2 ), float_3( -centerSpread, 0.0f, 0.0f ), float_3( 0, 0, 20 ), _spread );
	}

	// Copy particles to GPU memory.
	index< 1 > begin( 0 );
	concurrency::extent< 1 > end( _maxParticles );
	uploadData();
}

void NBodyAMPApp::uploadData()
{
	auto& task = _tasks[ _task ];
	copy( _particlesCpu.pos.begin(), _particlesCpu.pos.end(), task->dataOld->pos );
	copy( _particlesCpu.vel.begin(), _particlesCpu.vel.end(), task->dataOld->vel );
}

void NBodyAMPApp::setup()
{
	// NBody
	_nBody = std::make_unique< NBodyAMP >();

	// Tasks setup
	_tasks = createTasks( _maxParticles );

	// Particles setup
	loadParticles();

	// Parameters setup
	_params = { "Parameters", Vec2i( 225, 200 ) };
	_params.addParam( "Particles number", &_numParticles )
		.min( static_cast<float>( _minParticles ) )
		.max( static_cast<float>( _maxParticles ) )
		.step( static_cast<float>( _stepParticles ) );
	_params.addButton( "Reset particles", [ this ](){ loadParticles(); } );

	typedef std::codecvt_utf8< wchar_t > convert_type;
	std::wstring_convert< convert_type, wchar_t > converter;

	std::vector< std::string > accelerators( _tasks.size() );
	std::transform( _tasks.begin(), _tasks.end(), accelerators.begin(), [ &converter ]( const std::unique_ptr< TaskData >& task ){ return converter.to_bytes( task->accelerator.description ); } );
	_params.addParam( "Accelerator", accelerators, &_parameterTask );
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

void NBodyAMPApp::prepareSettings( Settings* settings )
{
	settings->setFrameRate( 60.0f );
}

void NBodyAMPApp::update()
{
	const auto elapsed = getElapsedSeconds();
	if( static_cast<size_t>( _parameterTask ) != _task )
	{
		_task = static_cast<size_t>( _parameterTask );
		uploadData();
	}
	
	auto& task = _tasks[ _task ];
	_nBody->integrate( *task, _numParticles, elapsed - _lastElapsed );
	
	std::swap( task->dataNew, task->dataOld );
	_fps = getAverageFps();
	_lastElapsed = elapsed;
}

void NBodyAMPApp::draw()
{
	// clear out the window with black
	gl::clear( Color( 0, 0, 0 ) );

	// draw particles
	auto& task = _tasks[ _task ];
	copy( task->dataOld->pos, _particlesCpu.pos.begin() );
	_mesh->bufferDynamicCustomAttributeData( 0u, _particlesCpu.pos );

	_shader->bind();
	gl::drawArrays( _mesh, 0, _numParticles );
	_shader->unbind();
	

	_params.draw();
}

CINDER_APP_NATIVE( NBodyAMPApp, RendererGl )
