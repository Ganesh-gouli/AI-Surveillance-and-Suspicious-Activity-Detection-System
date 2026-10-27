"""
SentinelVision AI - Results and Analysis REST API Router
Covers Academic Report Sections 7.1 to 7.9.
Provides Demonstration and Experimental results data, CSV exports, and live evaluation triggers.
"""

import os
from fastapi import APIRouter, Query, Response, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from typing import Optional, Dict, Any

from ...results.results_manager import global_results_manager, RESULTS_DIR

router = APIRouter(prefix="/results", tags=["Results & Analysis"])


@router.get("/summary")
def get_results_summary(mode: str = Query("demonstration", description="Result mode: demonstration or experimental")):
    """Returns complete summary of results across all sections 7.1 to 7.9."""
    mgr = global_results_manager
    return {
        "mode": mode,
        "training_results": mgr.get_model_training_results(mode),
        "evaluation_metrics": mgr.get_evaluation_metrics(mode),
        "confusion_matrix": mgr.get_confusion_matrix(mode),
        "training_curves": mgr.get_training_curves(mode),
        "environmental_results": mgr.get_environmental_results(mode),
        "hybrid_comparison": mgr.get_hybrid_model_comparison(mode),
        "system_interfaces": mgr.get_system_interfaces(),
        "performance_benchmarks": mgr.get_performance_benchmarks(mode),
        "traditional_comparison": mgr.get_traditional_comparison(mode)
    }


@router.get("/training")
def get_model_training_results(mode: str = Query("demonstration")):
    """Section 7.1: Model Training Results."""
    return global_results_manager.get_model_training_results(mode)


@router.get("/metrics")
def get_evaluation_metrics(mode: str = Query("demonstration")):
    """Section 7.2: Evaluation Metrics."""
    return global_results_manager.get_evaluation_metrics(mode)


@router.get("/confusion-matrix")
def get_confusion_matrix(mode: str = Query("demonstration")):
    """Section 7.3: Confusion Matrix Analysis."""
    return global_results_manager.get_confusion_matrix(mode)


@router.get("/curves")
def get_training_curves(mode: str = Query("demonstration")):
    """Section 7.4: Accuracy and Loss Curves."""
    return global_results_manager.get_training_curves(mode)


@router.get("/environmental")
def get_environmental_robustness(mode: str = Query("demonstration")):
    """Section 7.5: Environmental Robustness Analysis."""
    return global_results_manager.get_environmental_results(mode)


@router.get("/hybrid")
def get_hybrid_model_comparison(mode: str = Query("demonstration")):
    """Section 7.6: Hybrid Model Results."""
    return global_results_manager.get_hybrid_model_comparison(mode)


@router.get("/interfaces")
def get_system_interfaces():
    """Section 7.7: System Interface Results Documentation."""
    return {
        "interfaces": global_results_manager.get_system_interfaces(),
        "total_interfaces": len(global_results_manager.get_system_interfaces())
    }


@router.get("/benchmarks")
def get_performance_benchmarks(mode: str = Query("demonstration")):
    """Section 7.8: Hardware Performance Benchmarks."""
    return global_results_manager.get_performance_benchmarks(mode)


@router.get("/traditional")
def get_traditional_comparison(mode: str = Query("demonstration")):
    """Section 7.9: Comparison with Traditional Machine Learning Models."""
    return global_results_manager.get_traditional_comparison(mode)


@router.get("/export/csv/{section_id}")
def export_csv_artifact(section_id: str):
    """
    Downloads pre-generated CSV reports:
    - metrics: evaluation_metrics.csv
    - models: model_comparison.csv
    - benchmarks: performance_benchmarks.csv
    - history: training_history.csv
    """
    csv_map = {
        "metrics": "evaluation_metrics.csv",
        "models": "model_comparison.csv",
        "benchmarks": "performance_benchmarks.csv",
        "history": "training_history.csv"
    }

    filename = csv_map.get(section_id.lower())
    if not filename:
        raise HTTPException(status_code=404, detail=f"Unknown CSV artifact '{section_id}'. Options: metrics, models, benchmarks, history")

    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        global_results_manager.export_all_artifacts()

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=filename
    )


@router.post("/export/all")
def trigger_export_all():
    """Forces regeneration and export of all JSON and CSV artifacts."""
    files = global_results_manager.export_all_artifacts()
    return {
        "message": "All results artifacts successfully exported to results/ directory.",
        "files": files
    }


@router.post("/benchmarks/run")
def trigger_hardware_benchmark():
    """Triggers live hardware benchmark execution."""
    try:
        from ....training.system_performance_benchmarker import run_full_performance_benchmark
        payload = run_full_performance_benchmark()
        return {"status": "SUCCESS", "benchmark": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")
