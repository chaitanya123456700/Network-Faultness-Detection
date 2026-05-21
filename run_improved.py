"""
Complete Improved Pipeline for Network Fault Detection
Using NSL-KDD dataset with class balancing and improved architecture
"""
import os
import sys


def run_improved_pipeline():
    print("\n" + "="*70)
    print("  IMPROVED NETWORK FAULT DETECTION PIPELINE")
    print("  Using NSL-KDD Dataset with Class Balancing")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists('data/KDDTrain+.txt'):
        print("\n❌ ERROR: Dataset not found!")
        print("\nPlease download the NSL-KDD dataset:")
        print("1. Create data/ directory: mkdir -p data")
        print("2. Download files:")
        print("   wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt -O data/KDDTrain+.txt")
        print("   wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt -O data/KDDTest+.txt")
        return
    
    print("\n✅ Dataset files found!")
    
    # Step 1: Process data with improvements
    print("\n" + "="*70)
    print("[STEP 1/2] Processing NSL-KDD with Class Balancing")
    print("="*70)
    
    try:
        from data.data_loader import ImprovedNSLKDDLoader
        
        loader = ImprovedNSLKDDLoader(
            train_path='data/KDDTrain+.txt',
            test_path='data/KDDTest+.txt'
        )
        
        # Process with oversampling to balance classes
        G, train_snapshots, test_snapshots = loader.process_dataset(
            balance_method='oversample'
        )
        
        print("\n✅ Data processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error in data processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Train model
    print("\n" + "="*70)
    print("[STEP 2/2] Training Improved GAT Model")
    print("="*70)
    
    try:
        os.system('python train.py')
    except Exception as e:
        print(f"\n❌ Error in training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ├── data/network_graph.pkl         - Network topology")
    print("   ├── data/timeseries_data.pkl       - Training snapshots")
    print("   ├── data/test_data.pkl             - Test snapshots")
    print("   ├── outputs/best_model.pth         - Trained model")
    print("   ├── outputs/training_curves.png    - Training plots")
    print("   └── outputs/confusion_matrix.png   - Results visualization")
    
    print("\n📊 Key Improvements:")
    print("   ✅ Class balancing (oversampling minority classes)")
    print("   ✅ Better graph construction (protocol/service-based)")
    print("   ✅ More features (38 selected features)")
    print("   ✅ Weighted loss function")
    print("   ✅ Improved GAT architecture (4 layers, 128 hidden)")
    print("   ✅ Longer training (100 epochs)")
    
    print("\n🎯 Expected Results:")
    print("   • Test Accuracy: 75-85% (improved from 69%)")
    print("   • All classes detected (no 0% classes)")
    print("   • Balanced F1-scores across attack types")
    print("   • Better generalization")
    
    print("\n✅ Done! Check outputs/ folder for results.\n")


if __name__ == "__main__":
    run_improved_pipeline()