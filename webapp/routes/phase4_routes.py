from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_from_directory
from flask_login import login_required, current_user
import os
from datetime import datetime

from webapp.models import Dataset, Report, Analysis
from webapp.extensions import db

from engine.integration.analysis_runner import run_analysis
from engine.integration.report_builder import build_report, get_report

phase4_bp = Blueprint('phase4', __name__, template_folder='templates', url_prefix='/analysis')


def _user_datasets():
    return Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.uploaded_at.desc()).all()


@phase4_bp.route('/label-noise')
@login_required
def label_noise():
    datasets = _user_datasets()
    return render_template('analysis/label_noise.html', datasets=datasets)


@phase4_bp.route('/label-noise/run', methods=['POST'])
@login_required
def label_noise_run():
    dataset_id = request.form.get('dataset_id')
    target_col = request.form.get('target_col')
    ds = Dataset.query.get(dataset_id)
    if not ds:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('phase4.label_noise'))

    config = {'target_col': target_col}
    result = run_analysis(ds.upload_path, 'label_noise', config)
    if result.get('status') != 'ok':
        flash(f"Analysis failed: {result.get('error')}", 'danger')
        return redirect(url_for('phase4.label_noise'))

    dataset_meta = {'name': ds.original_name, 'shape': ''}
    report_id = build_report(result['results'], dataset_meta)

    # register DB records
    analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
    db.session.add(analysis)
    db.session.commit()

    return redirect(url_for('phase4.view_report', report_id=report_id))


@phase4_bp.route('/leakage')
@login_required
def leakage():
    datasets = _user_datasets()
    return render_template('analysis/leakage.html', datasets=datasets)


@phase4_bp.route('/leakage/run', methods=['POST'])
@login_required
def leakage_run():
    dataset_id = request.form.get('dataset_id')
    target_col = request.form.get('target_col')
    date_col = request.form.get('date_col')
    ds = Dataset.query.get(dataset_id)
    if not ds:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('phase4.leakage'))

    config = {'target_col': target_col, 'date_col': date_col}
    result = run_analysis(ds.upload_path, 'leakage', config)
    if result.get('status') != 'ok':
        flash(f"Analysis failed: {result.get('error')}", 'danger')
        return redirect(url_for('phase4.leakage'))

    report_id = build_report(result['results'], {'name': ds.original_name})
    analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('phase4.view_report', report_id=report_id))


@phase4_bp.route('/missing-data')
@login_required
def missing_data():
    datasets = _user_datasets()
    return render_template('analysis/missing_data.html', datasets=datasets)


@phase4_bp.route('/missing-data/run', methods=['POST'])
@login_required
def missing_data_run():
    dataset_id = request.form.get('dataset_id')
    target_col = request.form.get('target_col')
    ds = Dataset.query.get(dataset_id)
    if not ds:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('phase4.missing_data'))

    config = {'target_col': target_col}
    result = run_analysis(ds.upload_path, 'missing_data', config)
    if result.get('status') != 'ok':
        flash(f"Analysis failed: {result.get('error')}", 'danger')
        return redirect(url_for('phase4.missing_data'))

    report_id = build_report(result['results'], {'name': ds.original_name})
    analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('phase4.view_report', report_id=report_id))


@phase4_bp.route('/causal')
@login_required
def causal():
    datasets = _user_datasets()
    return render_template('analysis/causal.html', datasets=datasets)


@phase4_bp.route('/causal/run', methods=['POST'])
@login_required
def causal_run():
    try:
        dataset_id = request.form.get('dataset_id')
        treatment = request.form.get('treatment_col')
        outcome = request.form.get('outcome_col')
        nodes = request.form.get('nodes')
        edges = request.form.get('edges')
        ds = Dataset.query.get(dataset_id)
        if not ds:
            flash('Dataset not found.', 'warning')
            return redirect(url_for('phase4.causal'))

        config = {'treatment_col': treatment, 'outcome_col': outcome, 'nodes': nodes, 'edges': edges}
        result = run_analysis(ds.upload_path, 'causal', config)
        if result.get('status') != 'ok':
            flash(f"Analysis failed: {result.get('error')}", 'danger')
            return redirect(url_for('phase4.causal'))

        report_id = build_report(result['results'], {'name': ds.original_name})
        analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
        db.session.add(analysis)
        db.session.commit()
        return redirect(url_for('phase4.view_report', report_id=report_id))
    except Exception as e:
        current_app.logger.error(f"Causal analysis route error: {str(e)}")
        # Return JSON error as requested, although it might break form redirect
        return {"status": "error", "message": f"Server error during causal analysis: {str(e)}"}, 500


@phase4_bp.route('/explainability')
@login_required
def explainability():
    datasets = _user_datasets()
    return render_template('analysis/explainability.html', datasets=datasets)


@phase4_bp.route('/explainability/run', methods=['POST'])
@login_required
def explainability_run():
    dataset_id = request.form.get('dataset_id')
    target_col = request.form.get('target_col')
    ds = Dataset.query.get(dataset_id)
    if not ds:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('phase4.explainability'))

    config = {'target_col': target_col}
    result = run_analysis(ds.upload_path, 'explainability', config)
    
    # Even if failed, we want to see the logs/error in a report
    results_to_report = result.get('results')
    if result.get('status') != 'ok':
        # Create a dummy failure result if none exists
        if not results_to_report:
            results_to_report = {
                "module": "ExplainabilityEngine",
                "severity": "CRITICAL",
                "passed": False,
                "findings": {"error": result.get('error')},
                "log": result.get('log', []),
                "timestamp": datetime.now().isoformat()
            }
        flash(f"Analysis completed with issues: {result.get('error')}", 'warning')

    report_id = build_report(results_to_report, {'name': ds.original_name})
    analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('phase4.view_report', report_id=report_id))


@phase4_bp.route('/root-cause')
@login_required
def root_cause():
    datasets = _user_datasets()
    return render_template('analysis/root_cause.html', datasets=datasets)


@phase4_bp.route('/root-cause/run', methods=['POST'])
@login_required
def root_cause_run():
    dataset_id = request.form.get('dataset_id')
    target_col = request.form.get('target_col')
    ds = Dataset.query.get(dataset_id)
    if not ds:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('phase4.root_cause'))

    config = {'target_col': target_col}
    result = run_analysis(ds.upload_path, 'root_cause', config)
    
    results_to_report = result.get('results')
    if result.get('status') != 'ok':
        if not results_to_report:
            results_to_report = {
                "module": "RootCauseEngine",
                "severity": "CRITICAL",
                "passed": False,
                "findings": {"error": result.get('error')},
                "log": result.get('log', []),
                "timestamp": datetime.now().isoformat()
            }
        flash(f"Analysis completed with issues: {result.get('error')}", 'warning')

    report_id = build_report(results_to_report, {'name': ds.original_name})
    analysis = Analysis(user_id=current_user.id, dataset_name=ds.original_name, report_id=report_id, status='COMPLETED', config=config)
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('phase4.view_report', report_id=report_id))


@phase4_bp.route('/reports/phase4/<report_id>')
@login_required
def view_report(report_id):
    # Try to load the report data from the JSON file
    report_data = get_report(report_id)
    if not report_data:
        return ('Report not found', 404)
    
    # Render using the proper HTML template instead of raw file serving
    return render_template('reports/phase4_report.html', 
                           report_id=report_id, 
                           data=report_data['json'])
