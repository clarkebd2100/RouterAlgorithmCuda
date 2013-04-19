/******************|********************|*******************|******************|
 *																			   *
 *								ROUTER ALGORITHM							   *
 *																			   *
 *		PREPARED BY: 	Bennett D. Clarke									   *
 *						University of Illinois ( Chicago )					   *
 *						Chicago, Il.  60610									   *
 *		Date:			April 17, 2013									   	   *
 *																			   *
 *******************************************************************************
 *																			   *
 *	OBJECTIVE:	   	To use a data set of routing nodes and to use a parallel   *
 *					processor to run a single source shortest path algorithm   *
 *					to find the first hop node from the chosen source to any   *
 *					of the nodes on the routing topology.  The code is in      *
 *					Cuda.								                       * 							   		   
 *																			   *
 *	METHODOLOGY:	Use Dykstra's algorithm with uniform cost of 1 per router  *
 *					hop ( 1 per edge ).										   *
 ******************************************************************************** 
 *																				*
 *	DESCRIPTION OF VARIABLES													*
 *																				*
 *	edgeArrayParallel[ mVertex + nEdges ]	Large Array with All vertices		*
 *												and their outbound neighbors	*
 *	EdgeAPCount								Index of edgeArrayParallel for both *
 *												terminus of outbound edges and	*
 *												flags for end of linked list	*
 *	EdgeTable[ mVertices ]					Array of pointers; for each vertex	*
 *												the array entry points to a 	*
 *												linked list of outbound edges	*
 *												each of which is a neighbor of	*
 *												the indexed vertex				*
 *	edgeWeightParallel [ iEdge ] 			Weight set to 1000000 until reduced *
 *	leavingEdgeNo							Count of number of outbound edges   *
 *												from a Vertex					*
 *	processingNode[ ]															*				
 *	vertexArrayParallel[ iVertex ]			Each index represents corresponding	*
 *												vertex and it points to the		*
 *												index in the edgeArrayParallel  *
 *												where its first neighbor is 	*
 *												represented						*
 *							
 *******************************************************************************/
 
 
 
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cstdlib>
//#include <limits>
#include <cstdio>
#include <new>

//	CUDA INCLUDES
#include <cuda.h>

//  CUDA BY EXAMPLE CODE
#include "../common/book.h"


// DECLARATIONS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//#define N	10			// DEFINE FOR THREAD BLOCK NUMBER

using std::cout;
using std::endl;

typedef unsigned int vertex;

//CUDA KERNELS

__global__ void ssss_1 ( int N, bool * processingNode, float *  costToNode , 
			float * updatedCost, int * vertexArrayParallel, int * edgeArrayParallel,
			 float * edgeWeightParallel, int * accessedNode, int * parentArray  )
{
	int tid = blockIdx.x;
	int nid;
	if ( tid < N )
	{ 
		if( processingNode[ tid ] )
		{
			processingNode[ tid ] = false;  // Remove node from the exploring group
			
			// TEST IF ORIGINATING VERTEX HAS NEIGHBORS OTHER THAN ITS INBOUND
			// ROUTES FROM THE SOURCE  --  -1 MEANS NO OUTBOUND EDGES
	 		if ( vertexArrayParallel [ tid ]  != -1 ){
				
				// OUTBOUND EDGE TERMINUS IS NID; VAP[ TID ] GIVES THE INDEX IN EAP
				// WHERE THE OUTBOUND EDGE TERMINUS VERTEX ID CAN BE FOUND
				int eAPIndex = vertexArrayParallel[ tid ];
				nid = edgeArrayParallel [ eAPIndex ];			
				
				// THE ARRAY ENTRIES THAT CORRESPOND TO A LINKED LIST OF EDGES 
				// EMANATING FROM THE EXPANDING NODE WILL TERMINATE IN AN ARRAY ENTRY
				// EQUAL TO -1   -- EACH SEPARATE VALID NID REPRESENTS THE TERMINUS
				// OF A NEIGHBOR NODE  
 				while ( -1 != nid  ){
 					++accessedNode[ nid ];
					if ( updatedCost[ nid ] > costToNode[ tid ] + 
														edgeWeightParallel[ eAPIndex ] )
					{
						updatedCost[ nid ] = 
									costToNode [ tid ] + edgeWeightParallel[ eAPIndex ];
						parentArray[ nid ] = tid;
					}
					
					// INCREMENT THE INDEX TO THE EAP BY 1 TO GET NEXT OUTBOUND EDGE
					// TERMINUS - OTHERWISE KNOWN AS NEIGHBOR
 					//++(vertexArrayParallel[ tid ]);
 					++eAPIndex;
 					nid = edgeArrayParallel [ eAPIndex ];
 				} // END WHILE
				
				// FINISHED PROCESSING ALL OUTBOUND EDGES TO NEIGHBORS
				// RETURN DATA TO CPU is coded in cpu section
 				
 			}  // END if != NULL
		}	// END IF PROCEESINGNODE
	}	//	END IF TID < N
	

} // END KERNEL 1

__global__ void ssss_2 ( int N, bool * processingNode, float *  costToNode , 
			float * updatedCost, int * vertexArrayParallel, int * edgeArrayParallel,
			 							float * edgeWeightParallel  )
{
	int tid = blockIdx.x;
	if ( tid < N )
	{ 
 		if( costToNode[ tid ] > updatedCost[ tid ] )
 		{
 			costToNode[ tid ] = updatedCost[ tid ];
 			processingNode[ tid ] = true;
 		} // END UPDATE PATH TO NODE COST
 			else {
 			updatedCost[ tid ] = costToNode[ tid ];
 		}	
	} // END IF tid is valid
} // END KERNEL 2

// VARIABLES
bool undirectedGraphFlag ( false );


//	A LINE OF DATA FROM INPUT FILE CONSISTING OF 2 VERTICES AND THE COST OF A TRAVERSAL
//  FROM VERTEX A TO VERTEX B
struct DataLineIn{

	public:
	vertex vertexANo;
	vertex vertexBNo;
	float weight;
	
	DataLineIn( vertex initVertexANo, vertex initVertexBNo, float initWeight )
		: vertexANo( initVertexANo ), vertexBNo( initVertexBNo ), weight( initWeight ) 
	{}	


};

//	NODE CONSISTING OF A LINE OF DATA ( VERTEX A, VERTEX B, WEIGHT ) AND PTR
struct Node{

	DataLineIn 	data;
	Node *		next;
	
	Node( DataLineIn initData )
		: data ( initData )
	{
		next = NULL;
	}
	
	Node( DataLineIn initData, Node * initNext )
		: data ( initData ), next ( initNext )
	{}	
	
	Node( vertex initVertexANo, vertex initVertexBNo, float initWeight )
		: data( initVertexANo, initVertexBNo, initWeight), next ( NULL )
	{
		std::cout << "Detailed constructor" << std::endl;	
	}
	 
};

struct EdgeQueue{
	
		Node * 		head;
		Node * 		tail;
		
		EdgeQueue( )
		{
			head = NULL;
			tail = NULL;
		}

};

struct TreeNode{
	
	vertex 									vertexNo;
	float									distance;
	std::map < vertex, TreeNode * > 		childMapPtr;
	TreeNode * 								parent;
	
	TreeNode( vertex initVertexNo, float initDistance )
		: vertexNo ( initVertexNo ), distance ( initDistance )
	{
		parent = NULL;
	}	

};

struct Queue{

	TreeNode *				head;
	TreeNode * 				tail;
	
	Queue( )
	{
		head = NULL;
		tail = NULL;
	}
	
};

// 	VECTOR CONSISTING OF DATA IN ENTRIES ( VERTEX A, VERTEX B, WEIGHT )
std::vector < DataLineIn * > DataIn;


// FUNCTIONS


//	PROVIDES USAGE
static void show_usage(std::string fileName)
{
    std::cerr << "Running RouterAlgorithmArt.cu\n"	
    			"Usage: " << fileName << " <option(s)> DATA INPUT FILE  " << "Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-d,--data_file_path \tSpecify the data inputs path"
            << "\t-u,--undirected graph input option"
            << std::endl;
}




int main ( int argc, char ** argv ){

	if ( argc < 3 ) {
        show_usage( argv[ 0 ] );
        return 1;
    }
    std::string dataIn_Name = argv[ 1 ];
    for ( int i = 1; i < argc; ++i ) {
        std::string arg = argv[ i ];
        if ( (   arg == "-h" ) || ( arg == "--help" ) ) {
            show_usage( argv[ 0 ] );
            return 0;
        } else if ( arg == "-u" ){
        	undirectedGraphFlag = true;
        } else if ( (  arg == "-d") || ( arg == "--data_file_path" ) ) {
            if ( i + 1 < argc ) { 
                dataIn_Name = argv[ ++i ];
            } else {
                std::cerr << "--data_file_path option requires one argument." 
                			<< std::endl;
                return 1;
            }  
        }    
	}	// END FOR LOOP
	
	std::cout << "datafilename is: " << dataIn_Name << std::endl;
	unsigned int mV ( 0 );		// TO INPUT NUMBER OF VERTICES
	unsigned int nE ( 0 );		// TO INPUT NUMBER OF EDGES
	int a, b;					// INTEGER REPRESENTATION OF VERTICES
	double c;					// DOUBLE REPRESENTATION OF WEIGHT
	std::string line;
	std::ifstream dataIn;			// STREAM TO INPUT DATAFILE
	
	// PROCESS INPUT FILE INTO VECTOR OF ENTRIES (VERTEXA, VERTEXB, WEIGHT)
	dataIn.open ( dataIn_Name.c_str() );
	if ( dataIn.is_open() )
  	{
    	//	READ FIRST LINE OF DATA WITH SIZE OF GRAPH AND PRINT
    	dataIn >> mV >> nE;
    	
    	// UNDIRECTED GRAPH OPTION
    	if ( undirectedGraphFlag == true ){
    		nE *= 2;
    	}	
    	std::cout << "mVertices: " << mV << ", nEdges: " << nE << std::endl;
    	
    	// PROCESS BODY OF DATA
    	unsigned int dataCount ( 0 );
    	while ( dataCount < nE )
    	{
      		// READ A LINE OF DATA FROM FILE AND PRINT IT
      		dataIn >> a >> b >> c;
      		std::cout << "a: " << a << ", b: " << b << ", c: " << c << std::endl;
      		DataLineIn *ptr = new DataLineIn( a, b, c );
      		
      		// Enqueue new data element ptr to vector
      		DataIn.push_back( ptr );
      		
      		// UNDIRECTED GRAPH OPTION
      		if ( undirectedGraphFlag == true ) {
				cout << "a: " << b << ", b: " << a << ", c: " << c << endl;
				DataLineIn *ptr = new DataLineIn( b, a, c );
			
				// ENQUEUE
				DataIn.push_back( ptr );
				++dataCount;
      		}
      		
      		// INCREMENT FOR NEXT DATA LINE INPUT
      		++dataCount;
    	}
    	dataIn.close();
    	std::cout << "DataIn contains " << DataIn.size() << " records." << std::endl;
  	}
  	//	IF FILE FAILS TO OPEN
	else std::cout << "Unable to open file";
	
	// CREATE CONST UNSIGNED INT TO USE FOR DECLARATION OF ARRAYS
	// DOES NOT WORK HERE
	const unsigned int mVertices ( mV );		// NUMBER OF VERTICES IN GRAPH
	const unsigned int nEdges ( nE );			// NUMBER OF EDGES IN GRAPH
	
	// INITIALIZE 2D DYNAMIC ADJACENCY TABLE TO AN INFINITE ( 1000000 ) VALUE
//	std::cout << "Create adjacency table " << std::endl;
//	float **  adjacency = new float* [ mVertices ];
//	for ( unsigned int i = 0; i < mVertices; ++i ){
//		adjacency[ mVertices ] = new float[ mVertices ];
//	}	
// 	
// 	std::cout << "Initialize Adjacency Table " << std::endl;
// 	for ( unsigned int iCount = 0; iCount < mVertices; iCount++ ){
// 		for ( unsigned int jCount = 0; jCount < mVertices; jCount++ ){
// 			std::cout << "at iter i,j, " << iCount << "," << jCount << "," << std::flush;
// 			//std::cout	<< adjacency[ iCount ][ jCount ] << std::endl;
// 
// 			adjacency[ iCount ][ jCount ] = 1000000.0;
// 		}	
// 	}
	
	// CREATE EDGE TABLE AND INITIALIZE
	std::cout << "Create and initialize Edge Table " << " Size of node* "
				<< sizeof( Node ** ) << std::endl;
	Node ** EdgeTable = new Node * [ mVertices ];
//		std::cout << "Error in creating EdgeTable array! " << std::endl;
	
	// INITIALIZE ARRAY OF NODE POINTERS TO NULL
	// Node * EdgeTable [ mVertices ];
 	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
  		EdgeTable [ iCount ]  = NULL;
  	}	
	
	// LOAD GRAPH DATA INTO ADJACENCY MATRIX 
	std::cout << "Loading data into adjacency table " << std::cout;
	unsigned int kCount = 0;
// 	while( kCount < DataIn.size() ){
// 		adjacency[ DataIn[ kCount ]->vertexANo ][ DataIn[ kCount ]->vertexBNo ] = 
// 														DataIn[ kCount ]->weight;											
// 		++kCount;
// 	} // END WHILE
	
	// PRINT ADJACENCY TABLE
// 	std::cout << "\nAdjacency Matrix:" << std::endl;
// 	std::cout << "  ";
// 	for( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
// 		std::cout << "   v" << iCount;
// 	}
//	std::cout << "Not used with CUDA" << std::endl;
// 	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
// 		std::cout << "v" << iCount;
// 		for( unsigned int jCount = 0; jCount < mVertices; ++jCount ){
// 			if( adjacency[ iCount ][ jCount ] > 999999.9 )
// 				std::cout << "    " << "-" ;
// 			else
// 				std::cout << "    " << adjacency[ iCount ][ jCount ];
// 		}
// 		std::cout << std::endl;
// 	}
	
	kCount = 0;
	while ( kCount < DataIn.size() ){
			
		// EDGE TABLE  -- EACH DATA LINE OF THE INPUT ( AN EDGE ) IS PLACED IN A NODE
		// THE NODE IS THEN INSERTED IN A LINKED LIST THAT IS UNIQUE FOR EACH VERTEX
		// DATA RECORDS NUMBER IS KCOUNT; EDGETABLE ARRAY 1 FOR EACH VERTEX
		// NOTE AUTHORS USE A SINGLE EDGE ARRAY WHERE THE LINKED LIST ELEMENTS ARE
		// HELD AS CONTIGUOUS ARRAY MEMBERS
		DataLineIn dat = *DataIn[ kCount ];
		Node * nPtr = new Node( dat );
		Node * traveller;
		
		// IF THE EDGE TABLE ENTRY HAS NO LINKED LIST - ADD AS FIRST NODE
		if ( !EdgeTable[ DataIn[ kCount ]->vertexANo ] ){
			EdgeTable[ DataIn[ kCount ]->vertexANo ]  = nPtr;
		} else
		
		// ELSE ADD AT END OF LIST
		{
			traveller = EdgeTable[ DataIn[ kCount ]->vertexANo ];
			while ( traveller->next ){
				traveller = traveller->next;
			}
			traveller->next = nPtr;	
		}		// END IF-ELSE BLOCKS	
		++kCount;
	
	} // END WHILE
	
	//	PRINT EDGE TABLE
	std::cout << "Edge Table:" << std::endl;
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		std::cout << iCount << ":  " ;
		Node * traveller;
		traveller = EdgeTable[ iCount ];
		while  ( traveller ){
			std::cout << iCount << "->" << traveller->data.vertexBNo << "   " ;
			traveller = traveller->next;
		}
		std::cout << std::endl;
	}
	
	
	
	//////////////////// CREATE EDGE ADJACENCY ARRAY FOR CUDA /////////////////////
	
	// SIZE OF EXTENDED ADJACENCY TABLE IS SAME AS NUMBER OF DATA RECORDS
	int * vertexArrayParallel = new int [ mVertices ];  // Each index represents its 
												// respectively numbered vertex
												// the value is the EdgeArray index							
												// of the terminus vertex of
												// the first edge in the linked list
												// Represents edge: index->Edarray[value]
	// INITIALIZE THE VERTEXARRAYPARALLEL ENTRIES TO -1 WHICH WILL REPRESENT
	// A VERTEX WITHOUT OUTBOUND EDGES
	for ( unsigned int iVertex = 0; iVertex < mVertices; ++iVertex ){
		vertexArrayParallel [ iVertex ] = -1;
	}
	// PRINT OUT ARRAY VALUES
//	for ( unsigned int iVertex = 0; iVertex < mVertices; ++iVertex ){
//		std::cout << iVertex << ": " << vertexArrayParallel[ iVertex ] << std::endl;
//	}	
														
	// CREATE AN ARRAY THAT WILL PROVIDE ONE ENTRY FOR EVERY OUT EDGE AND EVERY VERTEX
	// INITIALIZE TO -1  -- -1 WILL BE A FLAG TO REPRESENT THE END OF A LINKED LIST 
	// OF OUTBOUND EDGES
	int * edgeArrayP = new int [ nEdges + mVertices ];
	float * edgeWeightParallel = new float [ nEdges + mVertices ];
	
	// INITIALIZE LARGE PARALLEL ARRAY ( edgeArrayP ) WITH -1
	unsigned int iEdge  ( 0 );
	for ( iEdge = 0; iEdge < (nEdges + mVertices); ++iEdge ){
		edgeArrayP[ iEdge ] = -1;
	}
	
	
	unsigned int i ( 0 );
	cout << "After initialization the edgeArrayP:" << endl;
	for( i = 0; i < (nEdges + mVertices); ++i ){
		std::cout << "edgeArrayP[ " << i << " ] : " << edgeArrayP[ i ] 
					<< ", " << std::endl;		
	}	
	
	// INITIALIZE ARRAY OF WEIGHTS TO INFINITE = 1000000
	// NOTE THAT INDICES MUST MATCH THOSE OF THE EDGEARRAYP SO THAT THERE ARE
	// EDGE + VERTICES ENTRIES AND SOME REMAIN 1000000 AS FLAGS
	for ( iEdge = 0; iEdge < ( nEdges + mVertices ); ++iEdge ){
		edgeWeightParallel [ iEdge ] = 1000000;
	}
	cout << "edgeWeightParallel[ each edge ] array as initialized " << endl;
	for( i = 0; i < ( nEdges + mVertices ); ++i ){
		std::cout << "edgeWeightParallel[ " << i << " ] : " << edgeWeightParallel[ i ] 
					<< ", " << std::endl;		
	}
	
	
	// LOAD DATA
	int EdgeAPCount ( 0 );				// Index in EdgeArrayParallel
	std::cout << "Loading data to parallel arrays" << std::endl;
	for ( unsigned int iVertex = 0; iVertex < mVertices; ++iVertex ){
		Node * traveller;
		traveller = EdgeTable[ iVertex ];  // Each table entry is head of linked list
		int leavingEdgeNo ( 0 );			// Initialize number of outbound edges from V
		
		// FOR EACH OUTBOUND EDGE IN A LINKED LIST - TRAVELLER POINTS TO THE EDGE
		// THAT EMANATES FROM VERTEX NUMBERED IVERTEX
		while ( traveller ){
			
			// LEADINGEDGENO == 0 INDICATES THAT THIS IS THE FIRST OUTBOUND EDGE
			// FROM A SPECIFIC EMANATING VERTEX
			if ( ! leavingEdgeNo )
			{	
				// USE POINTER FOR FIRST EDGE IN EACH VERTICE'S LINKED LIST
				std::cout << "iVertex: " << iVertex << std::endl;
				
				// SET POINTER AND ITERATE FOR FIRST ELEMENT IN LINKED LIST
				std::cout << "Head of a list: " << std::flush;
				//vertexArrayParallel [ iVertex ] = &edgeArrayParallel[ EdgeAPCount ];
				vertexArrayParallel [ iVertex ] = EdgeAPCount;
				std::cout << "vertexA<index> to edgeTable<index>: " << "Vertex no: " 							
							<< iVertex 
							<< " maps to " 
							<< vertexArrayParallel [ iVertex ] << " in the large array" 		
							<< std::endl;
				++leavingEdgeNo;
			}
			
			// RECORD B (A->B) VERTEX IN EDGEARRAY FOR ALL NODES
			edgeArrayP[ EdgeAPCount ] = traveller->data.vertexBNo;	
			edgeWeightParallel[ EdgeAPCount ] = traveller->data.weight;
			std::cout << "General node data points to vertex: " 
						<< "edgeArrayP[ " << EdgeAPCount << " ]: " 
						<< edgeArrayP[ EdgeAPCount ] << " weight is " 
						<< edgeWeightParallel[ EdgeAPCount ] << std::endl;
			++EdgeAPCount;
			
			
			// ITERATE ON TRAVELLER IN LINKED LIST
			traveller = traveller->next;
// 			for( unsigned int i = 0; i < EdgeAPCount; ++i ){
// 				std::cout << "index : " << i << " is " << edgeArrayP[ i ] 
// 					<< ", " << std::endl;
// 			}	
		}	// END WHILE LOOP
		// for loop will increment to next vertex
		// PROVIDE A FLAG ( -1 ) TO SEPARATE ELEMENTS BY THEIR INITIATION VERTEX
		// A OF ( A->B )
		cout << "\t\t\tSentinel value: " 	<< "edgeArrayP[ " << EdgeAPCount << " ]: " 
						<< edgeArrayP[ EdgeAPCount ] << std::endl;
		++EdgeAPCount;
	}  // END FOR LOOP
	// CHECK ARRAYS
	for( int i = 0; i < EdgeAPCount; ++i ){
		std::cout << "edgeArrayP[ " << i << " ] : " << edgeArrayP[ i ] 
					<< ", " << std::endl;
	}
	
	for( i = 0; i < ( EdgeAPCount ); ++i ){
		std::cout << "index : " << i << " is " << edgeWeightParallel[ i ] 
					<< ", " << std::endl;		
	}	
	
	
	std::cout << "Declaring device data structures" << std::endl;
	int * dev_vertexArrayParallel;
	int * dev_edgeArrayP;
	float * dev_edgeWeightParallel;
	
 	cudaMalloc( ( void** ) &dev_vertexArrayParallel, mVertices * sizeof( int * ) );
 	cudaMalloc( ( void** ) &dev_edgeArrayP, (nEdges + mVertices) * sizeof( int ) );
 	cudaMalloc( ( void** ) &dev_edgeWeightParallel, 
 										( nEdges + mVertices ) * sizeof( float ) );
	
	////////////////////////////////////////////////////////////////////////////////
	
	// TRANSFORM TO ARRAY OF VERTICES AT HEAD OF LINKED LISTS
	int singleSourceId ( -1 );
	std::cout << "Enter integer identifier of single source node: " << std::flush;

	std::cin >> singleSourceId;
	
	cout << "Received data: " << singleSourceId << endl;

	
	// CREATE A SINGLE SOURCE DATALINE NODE - THIS POINTS TO ITSELF WITH WEIGHT 0
	float sourceWeight ( 0 );
	//DataLineIn * initDataLine = new DataLineIn( singleSourceId, singleSourceId,
	//																 sourceWeight );
	Node * initNode = new Node( singleSourceId, singleSourceId, sourceWeight );
	std::cout << "SS.weight: " << initNode->data.weight << std::endl;
	
	///////////////// DATA STRUCTURES FOR PARALLEL PROCESSING /////////////////////////
	// CREATE A BOOL ARRAY TO HOLD FLAG FOR NODES BEING PROCESSED
	bool processingNode[ mVertices ];
	
	// CREATE A COST ARRAY AS THE COST OF REACHING EACH NODE FROM THE SOURCE
	float costToNode[ mVertices ];
	
	// CREATE A UPDATE COST ARRAY AS THE TEMPORARILY CALCULATED LEAST COST TO
	// EACH NODE FROM THE SOURCE
	
	float updatedCost[ mVertices ];
	
	// INITIALIZE THE ARRAYS
	//	FALSE FOR NOT CURRENTLY EXAMINED
	// 	1000000.00 TO APPROXIMATE INFINITE COST OF AN UNCONNECTED VERTEX
	for ( unsigned int ii = 0; ii < mVertices; ++ii ){
		processingNode [ ii ] = false;
		updatedCost[ ii ] = costToNode[ ii ] = 1000000.00;
	}
	
	for ( unsigned int ii = 0; ii < mVertices; ++ii ){
		cout << "costToNode[ " << ii << " ] : " << costToNode[ ii ] << endl;
	}
	
	// CREATE PARENT ARRAY TO INDICATE PARENT OF ATTACHED NODE// INITIALIZE TO -1
	int parentArray[ mVertices ];
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		parentArray[ iCount ] = -1;
	}	
	
	// CREATE CORRESPONDING ARRAYS FOR GPU AND CUDA
	
 	bool *dev_processingNode;
 	float *dev_costToNode;
 	float *dev_updatedCost;
 	int *dev_parentArray;
 	
// 	// ALLOCATE MEMORY ON GPU
 	cudaMalloc( ( void** ) &dev_processingNode, mVertices * sizeof( bool ) );
 	cudaMalloc( ( void** ) &dev_costToNode, mVertices * sizeof( float ) );
 	cudaMalloc( ( void** ) &dev_updatedCost, mVertices * sizeof( float ) );
 	cudaMalloc( ( void** ) &dev_parentArray, mVertices * sizeof( int ) );
	
	std::cout << "parallel processing arrays initialized" << std::endl;
	
	
	//////////////////////////  END COST ARRAYS //////////////////////////////////////
	
	// CREATE A ROOT NODE FOR USE IN THE EXPLORED QUEUE ( element is the source
	// parent is null )
	// TreeNode has elements: vertexNo, distance to source, ptr to map of children,
	// ptr to parent
	TreeNode * root = new TreeNode( singleSourceId, 0.0 );
	root->parent = NULL;
	
	// CREATE A MAP WITH VERTEX NO = KEY; TREE NODE * = VALUE
	std::map< vertex, TreeNode* > * mapPtr = new std::map<vertex, TreeNode* >();
	root->childMapPtr = *mapPtr;
	
	
	// SET EXPLORED NODE QUEUE
	// ANALOG TO GPU SETTING BOOL TO FALSE
	Queue *exploredPtr = new Queue();
	
	// INITIALIZE ARRAY TO HOLD DISTANCE FROM SOURCE TO EACH NODE ID
	// CPU ANALOG OF COSTTONODE
	float distanceToS [ mVertices ];
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		distanceToS[ iCount ] = -1.0;
	}
	
	// ENTER 0 DISTANCE FOR THE SOURCE NODE
	distanceToS[ singleSourceId ] = 0.0;	
	
	// SET ARRAY "PARENT NODE ARRAY' TO HOLD PTR TO PARENT'S TREENODE
	TreeNode * parentNodeArray[ mVertices ];
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		parentNodeArray[ iCount ] = NULL;
	}
	
	// SET PARENTARRAY FOR GPU CALCULATIONS
	parentArray[ singleSourceId ] = singleSourceId;
	
	//INITIALIZE THE EXPLORING QUEUE WITH THE SINGLE SOURCE
	EdgeQueue *exploringPtr = new EdgeQueue();
	exploringPtr->head = initNode;
	std::cout << "adding to exploring queue: " << exploringPtr->head->data.vertexANo
				<< std::endl;	
				
				
//////////////////////////// DIAGNOSTIC STRUCTURES /////////////////////////////////

	int accessedNode[ mVertices ];
	int accessedNodeReply[ mVertices ];
	int *dev_accessedNode;
	for( int iCount = 0; iCount < mVertices; ++iCount ) {
		accessedNode[ iCount ] = accessedNodeReply[ iCount ] = 0;
	}
	
	cudaMalloc( ( void** ) &dev_accessedNode, mVertices * sizeof( int ) );
	

						
//////////////////////// GPU INITIALIZATION WITH SINGLE SOURCE /////////////////////

 	processingNode[ singleSourceId ] = true;
 	costToNode[ singleSourceId ] = 0.0;
	updatedCost[ singleSourceId ] = 0.0;	
	
	
////////////////////////  START GPU PROCESSING LOOP ///////////////////////////////////
	
	cudaMemcpy( dev_vertexArrayParallel, vertexArrayParallel, 
				mVertices * sizeof( int  ), cudaMemcpyHostToDevice );					
	cudaMemcpy( dev_edgeArrayP, edgeArrayP, ( mVertices + nEdges ) * sizeof( int ),
											cudaMemcpyHostToDevice );
	cudaMemcpy( dev_edgeWeightParallel, edgeWeightParallel, 
				( mVertices + nEdges ) * sizeof( float ), cudaMemcpyHostToDevice );	

	bool flagContinue ( true );
	while ( flagContinue ){

		// COPY INITIAL DATA TO GPU
		cudaMemcpy( dev_processingNode, processingNode, mVertices * sizeof( bool ),
										cudaMemcpyHostToDevice );
		cudaMemcpy( dev_costToNode, costToNode, mVertices * sizeof( float ),
										cudaMemcpyHostToDevice );	
		cudaMemcpy( dev_updatedCost, updatedCost, mVertices * sizeof( float ),
										cudaMemcpyHostToDevice );
		cudaMemcpy( dev_parentArray, parentArray, mVertices * sizeof( int ), 					
														cudaMemcpyHostToDevice );										
										
		//	DIAGNOSTIC FUNCTION
		cudaMemcpy( dev_accessedNode, accessedNode, 
				mVertices * sizeof( int  ), cudaMemcpyHostToDevice );	
	
				
		cout << "Copied data to gpu " << endl;
						

		// EXECUTE KERNEL ON GPU						
		ssss_1<<< mVertices, 1>>> ( mVertices, dev_processingNode, dev_costToNode, 
							dev_updatedCost, dev_vertexArrayParallel, dev_edgeArrayP, 
							dev_edgeWeightParallel, dev_accessedNode, dev_parentArray );	
															
		cout << "Processed kernel1 " << endl;
		

	
		// RETURN PROCESSED DATA FROM GPU
		HANDLE_ERROR( cudaMemcpy( updatedCost, dev_updatedCost, 
											mVertices * sizeof( float ),
											cudaMemcpyDeviceToHost ) );
											
		cout << "copied updatedCost " << endl;
//		exit ( 0 );
											
		cudaMemcpy( processingNode, dev_processingNode, mVertices * sizeof( bool ),
											cudaMemcpyDeviceToHost );
		cudaMemcpy( parentArray, dev_parentArray, mVertices * sizeof( int ),
											cudaMemcpyDeviceToHost );											
											
		// DIAGNOSTIC STRUCTURE
		cudaMemcpy( accessedNodeReply, dev_accessedNode, mVertices* sizeof( int ),
											cudaMemcpyDeviceToHost );
											
																									
	/////////////////////    END KERNEL 1   ///////////////////////////////////////////	

	cout << "End kernel 1" << endl;
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout << "processing [ " << iCount << " ] : " << processingNode[ iCount ] << endl;
	}
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout << "updatedCost [ " << iCount << " ] : " << updatedCost[ iCount ] << endl;
	}
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout << "costToNode [ " << iCount << " ] : " << costToNode[ iCount ] << endl;
	}
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout << "accessedNodeReply [ " << iCount << " ] : " << accessedNodeReply[ iCount ]
				<< endl;
	}			
	
//	exit ( 0 );

	///////////////////// KERNEL 2 ////////////////////////////////////////////////////

	
		// COPY INITIAL DATA TO GPU
		cudaMemcpy( dev_processingNode, processingNode, mVertices * sizeof( bool ),
										cudaMemcpyHostToDevice );
		cudaMemcpy( dev_costToNode, costToNode, mVertices * sizeof( float ),
										cudaMemcpyHostToDevice );	
		cudaMemcpy( dev_updatedCost, updatedCost, mVertices * sizeof( float ),
										cudaMemcpyHostToDevice );	
	
	
		// EXECUTE KERNEL ON GPU						
		ssss_2<<< mVertices, 1>>> ( mVertices, dev_processingNode, dev_costToNode, 
							dev_updatedCost, dev_vertexArrayParallel, dev_edgeArrayP, 
															dev_edgeWeightParallel );
	
		// RETURN PROCESSED DATA FROM GPU
		cudaMemcpy( costToNode, dev_costToNode, mVertices * sizeof( float ),
											cudaMemcpyDeviceToHost );
		cudaMemcpy( updatedCost, dev_updatedCost, mVertices * sizeof( float ),
											cudaMemcpyDeviceToHost );
		cudaMemcpy( processingNode, dev_processingNode, mVertices * sizeof( bool ),
											cudaMemcpyDeviceToHost );
//		cudaMemcpy( parentArray, dev_parentArray, mVertices * sizeof( int ),
//											cudaMemcpyDeviceToHost );										
											
		cout << "End kernel2" << endl;

	////////////////////////// END KERNEL 2 ////////////////////////////////////////////
	
		cout << "processingNode after kernel2: " << endl;
		for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
			cout << "processing [ " << iCount << " ] : " << processingNode[ iCount ] 
				<< endl;
		}
		for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
			cout << "updatedCost [ " << iCount << " ] : " << updatedCost[ iCount ] 
				<< endl;
		}
		for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
			cout << "costToNode [ " << iCount << " ] : " << costToNode[ iCount ] 
				<< endl;
		}
		for ( unsigned int iCount = 0; iCount < ( mVertices + nEdges ); ++iCount ){
			cout << "edgeWeightParallel [ " << iCount << " ] : " 
				<< edgeWeightParallel[ iCount ] << endl;
		}
		
		for ( unsigned int i = 0; i < mVertices; ++i ){
			if( processingNode[ i ] ) { 
				flagContinue = true; 
				goto repeatLoopTrue; 
			}
		}
		flagContinue = false;
	repeatLoopTrue:
		continue;																		
	}	// END GPU PROCESSING LOOP
	
	// PRINT RESULTS OF GPU PROCESSING
	cout << "Parent array: " << endl;
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout <<  "parentArray[ " << iCount << " ] : " << parentArray[ iCount ] << endl;
	}
	
	cout << "Destination Node\tDistance" << endl;
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		cout << "\t " << iCount << ":\t\t\t" << costToNode[ iCount ] << endl;
	}
	
	cout << "Finished distance" << endl;
	
	int firstHopArray[ mVertices ];
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		firstHopArray[ iCount ] = -1;
	}
	
	cout << "initialized firstHop" << endl;	
	int parent ( -1 );
	int intervenor ( -1 );
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
	
		cout << "FirstHop: " << iCount << endl;
		parent = -1;
		intervenor = -1;
		
		// FOR SOURCE NODE
		if( parentArray[ iCount ] == iCount ){
			firstHopArray[ iCount ] = iCount;
			continue;
		} 
		
		// FOR ISOLATED NODE
		if( parentArray[ iCount ] == -1 ){
			cout << iCount << " is isolated from the source node" << endl;
			continue;
		}
		
		// FOR ALL NON-SOURCE VERTICES - BACK TRACK UNTIL SINGLESOURCEID IS REACHED
		cout << "Not source: " << iCount << endl;
		parent = parentArray[ iCount ];
		intervenor = iCount;
		while ( parent != singleSourceId ){
			cout << iCount << " has parent: " << parent << endl;
			intervenor = parent;
			parent = parentArray[ parent ];
		}
		firstHopArray[ iCount ] = intervenor;
		
	}	// END FOR LOOP
	
	cout << "Destination Node\tFirst Hop Router" << endl;
	for( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		if( firstHopArray [ iCount ] == -1 )
		{
			continue;
		}	
		cout << "\t " << iCount << ":\t\t\t" << firstHopArray[ iCount ] << endl;
	}
		
	exit ( 0 );
	
std::cout << exploringPtr->head << std::endl;

std::cout << exploringPtr->head->data.vertexANo << std::endl;
std::cout << exploringPtr->head->data.weight << std::endl;
	
int xCount = 0;
// EXPLORINGPTR->HEAD IS THE NODE THAT IS BEING SEARCHED FOR THE LEAST COST NEXT HOP
while( exploringPtr->head ){
	// FIND CLOSEST NODE
	Node * trav;
	trav = exploringPtr->head;
	Node * followNode;
	Node * preMinNode;
	followNode = exploringPtr->head;	// PREVIOUS NODE ALLOWS LINK BACK
	
	// APPROXIMATE INFINITE DISTANCE ( COST ) WEIGHT AS 1000000.0 
	// FOR EACH NEW SEARCH FOR NEXT SHORTEST THROUGH LINK TO SOURCE
	float minDist ( 1000000.0 );
	Node * minNode; 
	while ( trav ){
		std::cout << "Checking for min with edge: " << trav->data.vertexANo 
					<< "->" << trav->data.vertexBNo << std::endl;
		if( distanceToS[ trav->data.vertexANo ] < 0.0 ){
			// ERROR CHECK
			std::cout << "ERROR - Negative distance" << std::endl;
		} else {
			// CHECK SOURCE-TO-TRAV + TRAV-TO-TEST EDGE FOR MIN TOTAL COST
			 if ( trav->data.weight + distanceToS[trav->data.vertexANo ] < minDist ) {
				minDist = trav->data.weight + distanceToS[ trav->data.vertexANo ];
				std::cout << "New min is Edge : " << trav->data.vertexANo << "->" 
						<< trav->data.vertexBNo << std::endl;
				std::cout << "weight: " << trav->data.weight << std::endl;		
				minNode = trav;
				preMinNode = followNode;
			 }
		 }
		 // INCREMENT ITERATORS SO THAT FOLLOWNODE IS IMMEDIATELY BEHIND trav
		 // TRAV ITERATES THROUGH ALL NODES THAT ARE IN EXPLORING QUEUE
		 // EXPLORING QUEUE = 
		 // NODE THAT ARE BEING CHECKED AS LINKS BUT CHECKS ARE NOT COMPLETE
		 if ( trav != exploringPtr->head )	
		 	followNode = trav;
		trav = trav->next; 
		std::cout << "end while loop: " << trav << std::endl;
	} // END WHILE LOOP TO SEARCH FOR THE NEXT SHORTEST THROUGH LINK TO THE SOURCE
	
	std::cout << "closest uncatalouged route source to node: " << std::flush;
	std::cout	<< minNode->data.vertexANo << "->" << std::flush;
	std::cout 	<< minNode->data.vertexBNo << ", distance: " << std::flush;
	std::cout			<< minDist << "!" << std::endl;	
	
	// REMOVE CLOSEST NODE FROM EXPLORING QUEUE
	if ( minNode == exploringPtr->head ){
		exploringPtr->head = exploringPtr->head->next;
	} else {
		preMinNode->next = minNode->next;
	}
	if ( minNode == exploringPtr->tail ){
		exploringPtr->tail = preMinNode;
	}	
	std::cout << "Deleted from exploring queue edge: " <<  minNode->data.vertexANo 
				<< "->" << minNode->data.vertexBNo << std::endl;
	
	// PURGE ALL INSTANCES OF REMOVED NODE (THEY ARE GREATER DISTANCE )
	trav = exploringPtr->head;
	followNode = exploringPtr->head;
	while( trav ){
		if ( trav->data.vertexBNo == minNode->data.vertexBNo ){
			if( trav == exploringPtr->head ){
				exploringPtr->head = exploringPtr->head->next;
			} else {
				followNode->next = trav->next;			
			}
			if ( trav == exploringPtr->tail ){
				exploringPtr->tail = followNode;
			}		
		}
		// INCREMENT ITERATORS SO THAT FOLLOWNODE IS IMMEDIATELY BEHIND trav
		 if ( trav != exploringPtr->head )	
		 	followNode = trav;
		trav = trav->next;
	}				

	// ADD REMOVED NODE TO THE EXPLORED QUEUE
		// IF EXPLORED NODE IS EMPTY MAKE THE FIRST NODE (SINGLE SOURCE ) ROOT
	if ( !exploredPtr->head ){
		exploredPtr->head = root;
		exploredPtr->tail = root;
		root->childMapPtr[ singleSourceId ] = root;
		std::cout << "Adding to explored queue root: " << exploredPtr->head->vertexNo
				<< std::endl;
				
		// SET ( LEAVE ) PARENTNODE QUEUE TO NULL
		parentNodeArray[ exploredPtr->head->vertexNo ] = root;
	}
		// IF EXPLORED NODE IS NOT EMPTY ADD A NEW NODE
		// ADD NEWEST NODE TO TREE
	else {
		TreeNode * nwNodePtr = 
						new TreeNode( minNode->data.vertexBNo, minNode->data.weight );
		
		// UPDATE PARENTNODEARRAY FOR NEW EXPLORED NODE
		parentNodeArray[ nwNodePtr->vertexNo ] = nwNodePtr;

		// FIND NWNODEPTR'S PARENT NODE FROM PARENTNODEARRAY AND 
		// ADD PARENT PTR TO NEW NODE, THEN ADD CHILD PTR TO PARENT
		// -node of parent
		nwNodePtr->parent = parentNodeArray[ minNode->data.vertexANo ];
		std::cout << "Parent Node is: " << nwNodePtr->parent->vertexNo << std::endl;
		// Parent adds ptr to child (new addition)
		nwNodePtr->parent->childMapPtr[ nwNodePtr->vertexNo ] = nwNodePtr;
		std::cout << "parent: " << nwNodePtr->parent->vertexNo << " has child in node: "
					<< nwNodePtr->parent->childMapPtr[ nwNodePtr->vertexNo ]->vertexNo
					<< std::endl;
		TreeNode * tnTrav = nwNodePtr->parent;			
		while( tnTrav->parent  ){
			tnTrav->parent->childMapPtr[ nwNodePtr->vertexNo ] = tnTrav;
			std::cout << "parent: " << tnTrav->parent->vertexNo 
					<< "has child on route to " << nwNodePtr->vertexNo << " in node: " 
					<< tnTrav->vertexNo << std::endl;
			tnTrav = tnTrav->parent;
		}
		
		// minNode->data.vertexANo as treenode.vertexNo
	} // END IF ELSE BLOCK
	
	// PRINT TREE
	std::cout << "root: " << root->vertexNo << std::endl;
	//int iCount = 0;
	
	// SET DISTANCE OF NODE ADDED TO EXPLORED QUEUE
	distanceToS[ minNode->data.vertexBNo ] = minDist;
	std::cout << "Setting total distance for node: " << minNode->data.vertexBNo
				<< " at " << minDist << std::endl;
				
	// PRINT EXPLORING QUEUE
	std::cout << "Exploring Queue after purge before expansion: ";
	trav = exploringPtr->head;
	if ( trav ){
		while ( trav ){
			std::cout << trav->data.vertexBNo << ", " ;
			trav = trav->next;
		}
	} else {
		std::cout << "empty queue";
	}		
	std::cout << std::endl;				
	
	// EXPAND CHOSEN NODE IN EXPLORING QUEUE AND ADD TO EXPLORING NODE QUEUE	
	// ADD ALL NODES REACHABLE FROM MINNODE (BUT NOT IN EXPLORED QUEUE )
	// TO THE EXPLORING QUEUE
	// minNode->data

	trav = EdgeTable[ minNode->data.vertexBNo ];
	while( trav ){
		if ( distanceToS[ trav->data.vertexBNo ] < 0.0 ){
			std::cout << "have entry(ies) to add" << std::endl;
			if ( !exploringPtr->head ){
				exploringPtr->head = trav;
				exploringPtr->tail = trav;
				std::cout << "Adding to queue" << std::endl;
			} else {
				exploringPtr->tail->next = trav;
				exploringPtr->tail = trav;
			}
			std::cout << "edge: " << trav->data.vertexANo << "->" 
						<< trav->data.vertexBNo << std::endl;			
		}			
		trav = trav->next;		
	}
	// PRINT EXPLORING QUEUE
	std::cout << "Exploring Queue: ";
	trav = exploringPtr->head;
	while ( trav ){
		std::cout << trav->data.vertexANo << "->" << trav->data.vertexBNo << ", " ;
		trav = trav->next;
	}
	std::cout << std::endl;	
	
	// PRINT DISTANCE ARRAY
	std::cout << "\nNode\tDistance From Source" << std::endl;
	for( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		std::cout << iCount << ": \t\t" << distanceToS[ iCount ] << std::endl;
	}	
	
	// PURGE ALL INSTANCES OF REMOVED NODE (THEY ARE GREATER DISTANCE )
	trav = exploringPtr->head;
	followNode = exploringPtr->head;
	while( trav ){
		if ( distanceToS[ trav->data.vertexBNo ] >= 0.0 ){
			if( trav == exploringPtr->head ){
				exploringPtr->head = exploringPtr->head->next;
			} else {
				followNode->next = trav->next;			
			}
			if ( trav == exploringPtr->tail ){
				exploringPtr->tail = followNode;
			}		
		}
		// INCREMENT ITERATORS SO THAT FOLLOWNODE IS IMMEDIATELY BEHIND trav
		 if ( trav != exploringPtr->head )	
		 	followNode = trav;
		trav = trav->next;
	}
++xCount;
} // END WHILE	

	std::cout << "First hop router list: \n" 
				<< "Destination Node	First Hop Node" << std::endl;
	for ( unsigned int iCount = 0; iCount < mVertices; ++iCount ){
		std::cout << "        " << iCount << "                     " 
				<< root->childMapPtr[ iCount ]->vertexNo << std::endl; 	
	
	}
	
	// DELETE POINTERS TO DYNAMIC OBJECTS

	cudaFree ( dev_vertexArrayParallel );
	cudaFree ( dev_edgeArrayP );
	cudaFree ( dev_edgeWeightParallel );
	cudaFree ( dev_processingNode );
 	cudaFree ( dev_costToNode );
 	cudaFree ( dev_updatedCost );
 	cudaFree ( dev_parentArray );
 	cudaFree ( dev_accessedNode );
 	

 	delete exploringPtr;
 	delete exploredPtr;
 	delete mapPtr;
 	delete root;
 	delete initNode;
 	
 	
 	delete [] 	edgeWeightParallel;
 	delete []	edgeArrayP;
 	delete [] 	vertexArrayParallel;
 	delete [] 	EdgeTable;
	
//	for ( unsigned int i = 0; i < mVertices; ++i ){
//		delete [] adjacency[ i ];
//	}
//	delete [] adjacency;
	
			
	return 0;

}
